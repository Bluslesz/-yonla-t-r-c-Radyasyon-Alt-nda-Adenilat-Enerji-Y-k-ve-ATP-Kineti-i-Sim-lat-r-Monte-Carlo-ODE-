
from __future__ import annotations

import argparse
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class ModelParameters:
    a_hit: float = 1.3
    a_rdem: float = 2.4
    a_rd: float = 0.5
    q_r: float = 0.04
    alpha_0: float = 22.9
    alpha_1: float = 79.4
    alpha_2: float = 0.0832
    beta_1: float = 0.08
    beta_2: float = 50.0

    L: float = 5.0e6
    k_s: float = 0.1
    sigma_m: float = 100.0
    k_r: float = 0.05
    c: float = 0.01
    eps: float = 0.1
    sigma_1_0: float = 0.5

    y0: tuple[float, float, float] = (2.5, 0.25, 0.02)
    simulation_end: float = 600.0
    steady_start: float = 300.0


PARAMETERS = ModelParameters()


def validate_dose(dose: float, name: str = "D") -> float:
    if not np.isfinite(dose) or dose < 0.0:
        raise ValueError(f"{name} sifirdan buyuk veya sifir olmalidir; gelen degil: {dose}")
    return float(dose)


def monte_carlo_damage_fraction(
    dose: float,
    n_cells: int = 2000,
    priming_dose: float = 0.0,
    delta_t: float = 4.0,
    seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """
    Verilen doz icin hucre basina etkili hasar yukunu Monte Carlo ile hesaplar.

    Bonus: (ortalama stres, ortalamaya ait standart hata, hucre stres skorlari).
    """
    dose = validate_dose(dose)
    priming_dose = validate_dose(priming_dose, "priming_dose")
    if n_cells < 100:
        raise ValueError("n_cells en az 100 olmalidir.")
    if not np.isfinite(delta_t) or delta_t < 0.0:
        raise ValueError("delta_t sifirdan buyuk veya sifir olmalidir.")
    if seed < 0:
        raise ValueError("seed negatif olamaz.")

    rng = np.random.default_rng(seed)
    p_hit = 1.0 - np.exp(-PARAMETERS.a_hit * dose)
    p_rdem = 1.0 - np.exp(-PARAMETERS.a_rdem * dose)
    p_rd = 1.0 - np.exp(-PARAMETERS.a_rd * dose)
    p_bystander = PARAMETERS.beta_1 * (1.0 - np.exp(-PARAMETERS.beta_2 * dose))
    p_bystander = float(np.clip(p_bystander, 0.0, 1.0))

    if priming_dose > 0.0:
        p_adaptive = (
            PARAMETERS.alpha_0
            * priming_dose**2
            * delta_t**2
            * np.exp(
                -PARAMETERS.alpha_1 * priming_dose
                - PARAMETERS.alpha_2 * delta_t
            )
        )
        p_adaptive = float(np.clip(p_adaptive, 0.0, 0.45))
    else:
        p_adaptive = 0.0

    hits = rng.random(n_cells) < p_hit
    direct_damage = hits & (rng.random(n_cells) < p_rdem)
    direct_death = hits & (rng.random(n_cells) < p_rd)
    bystander_stress = (~hits) & (rng.random(n_cells) < p_bystander)
    repaired = (direct_damage | bystander_stress) & (
        rng.random(n_cells) < (PARAMETERS.q_r + p_adaptive)
    )

    stress_scores = (
        1.0 * direct_death.astype(float)
        + 0.65
        * (direct_damage & ~repaired & ~direct_death).astype(float)
        + 0.25 * (bystander_stress & ~repaired).astype(float)
    )
    mean = float(np.mean(stress_scores))
    std_error = float(np.std(stress_scores, ddof=1) / np.sqrt(n_cells))
    return mean, std_error, stress_scores


def adenylate_kinetics_ode(
    _time: float,
    state: Sequence[float],
    stress_load: float,
) -> list[float]:
    """Goldbeter PFK ve adenilat kinaz bagimli adenilat turusleri."""
    if len(state) != 3:
        raise ValueError("ODE durumu [ATP, ADP, AMP] seklinde 3 elemanli olmalidir.")

    atp, adp, amp = np.maximum(np.asarray(state, dtype=float), 1e-8)
    alpha = atp / PARAMETERS.k_r
    gamma = adp / PARAMETERS.k_r

    term_r = (
        (alpha / (PARAMETERS.eps + 1.0))
        * (1.0 + alpha / (PARAMETERS.eps + 1.0))
        * (1.0 + gamma) ** 2
    )
    denominator = (
        PARAMETERS.L * (1.0 + alpha * PARAMETERS.c) ** 2
        + (1.0 + gamma) ** 2
        * (1.0 + alpha / (PARAMETERS.eps + 1.0)) ** 2
    )
    denominator = max(float(denominator), np.finfo(float).tiny)
    phi = float(term_r / denominator)

    sigma_eff = PARAMETERS.sigma_1_0 * np.exp(-2.8 * stress_load)
    k_repair_demand = 0.015 + 0.085 * stress_load
    k_ad = 2.5
    v_ad = k_ad * (adp**2 - 0.85 * atp * amp)
    pfk_flux = PARAMETERS.sigma_m * PARAMETERS.k_r * phi * 0.012

    d_atp = (
        sigma_eff * PARAMETERS.k_r
        - pfk_flux
        - k_repair_demand * atp
        + v_ad
    )
    d_adp = (
        pfk_flux
        + k_repair_demand * atp
        - 0.04 * (1.0 - 0.7 * stress_load) * adp
        - 2.0 * v_ad
    )
    d_amp = v_ad - 0.005 * amp
    return [float(d_atp), float(d_adp), float(d_amp)]


def solve_adenylate_kinetics(
    stress_load: float,
    t_eval: np.ndarray,
) -> Any:
    stress_load = float(np.clip(stress_load, 0.0, 1.0))
    if t_eval.size == 0:
        raise ValueError("t_eval bos olamaz.")

    solution = solve_ivp(
        adenylate_kinetics_ode,
        (0.0, float(PARAMETERS.simulation_end)),
        PARAMETERS.y0,
        args=(stress_load,),
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-7,
        atol=1e-9,
        max_step=2.0,
    )
    if not solution.success:
        raise RuntimeError(f"ODE cozumleyici hatasi: {solution.message}")
    if solution.y.shape[1] != t_eval.size:
        raise RuntimeError("ODE cozumu beklenen zaman noktalarinin tamaminda basarili olmadi.")
    if not np.all(np.isfinite(solution.y)):
        raise RuntimeError("ODE cozumu sonlu olmayan veya NaN degerler uretildi.")
    return solution


def adenylate_fractions(solution: Any) -> tuple[float, float, float, float]:
    atp = np.maximum(solution.y[0], 0.0)
    adp = np.maximum(solution.y[1], 0.0)
    amp = np.maximum(solution.y[2], 0.0)

    atp_ss = float(np.mean(atp))
    adp_ss = float(np.mean(adp))
    amp_ss = float(np.mean(amp))
    total = atp_ss + adp_ss + amp_ss
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("Adenilat mol kesirleri icin gecerli toplam bulunamadi.")

    atp_fraction = atp_ss / total
    adp_fraction = adp_ss / total
    amp_fraction = amp_ss / total
    aec = (atp_ss + 0.5 * adp_ss) / total
    return aec, atp_fraction, adp_fraction, amp_fraction


def simulate_dose_response(
    doses: np.ndarray,
    priming_dose: float = 0.0,
    n_cells: int = 3000,
    seed: int = 42,
    uncertainty_replicates: int = 5,
    progress_callback: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    priming_dose = validate_dose(priming_dose, "priming_dose")
    if uncertainty_replicates < 1:
        raise ValueError("uncertainty_replicates en az 1 olmalidir.")

    aec_means: list[float] = []
    aec_stds: list[float] = []
    atp_means: list[float] = []
    adp_means: list[float] = []
    amp_means: list[float] = []
    steady_eval = np.linspace(PARAMETERS.steady_start, PARAMETERS.simulation_end, 150)

    for dose_index, dose in enumerate(doses):
        if progress_callback is not None:
            progress_callback(dose_index, dose, 0)

        replicate_aec: list[float] = []
        replicate_atp: list[float] = []
        replicate_adp: list[float] = []
        replicate_amp: list[float] = []

        for replicate in range(uncertainty_replicates):
            replicate_seed = seed + dose_index * 1000 + replicate
            stress_mean, _, _ = monte_carlo_damage_fraction(
                float(dose),
                n_cells=n_cells,
                priming_dose=priming_dose,
                seed=replicate_seed,
            )
            solution = solve_adenylate_kinetics(stress_mean, steady_eval)
            aec, atp_fraction, adp_fraction, amp_fraction = adenylate_fractions(solution)
            replicate_aec.append(aec)
            replicate_atp.append(atp_fraction)
            replicate_adp.append(adp_fraction)
            replicate_amp.append(amp_fraction)
            if progress_callback is not None:
                progress_callback(dose_index, dose, replicate + 1)

        aec_means.append(float(np.mean(replicate_aec)))
        aec_stds.append(float(np.std(replicate_aec, ddof=1)) if uncertainty_replicates > 1 else 0.0)
        atp_means.append(float(np.mean(replicate_atp)))
        adp_means.append(float(np.mean(replicate_adp)))
        amp_means.append(float(np.mean(replicate_amp)))

    return (
        np.asarray(aec_means),
        np.asarray(aec_stds),
        np.asarray(atp_means),
        np.asarray(adp_means),
        np.asarray(amp_means),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Doz bagimli AEC egri ve adenilat havuz simulasyonu"
    )
    parser.add_argument("--dose-min", type=float, default=0.0)
    parser.add_argument("--dose-max", type=float, default=5.0)
    parser.add_argument("--dose-points", type=int, default=26)
    parser.add_argument("--priming-dose", type=float, default=0.05)
    parser.add_argument("--n-cells", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--uncertainty-replicates", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--no-show", action="store_true", help="Grafigi ekranda acma")
    parser.add_argument("--headless", action="store_true", help="Grafik arayuzu acmadan calistir")
    return parser


def validate_cli_args(args: argparse.Namespace) -> np.ndarray:
    args.dose_min = validate_dose(args.dose_min, "dose_min")
    args.dose_max = validate_dose(args.dose_max, "dose_max")
    args.priming_dose = validate_dose(args.priming_dose, "priming_dose")
    if args.dose_max < args.dose_min:
        raise ValueError("dose_max dose_min degerinden buyuk olmalidir.")
    if args.dose_points < 2:
        raise ValueError("dose_points en az 2 olmalidir.")
    if args.n_cells < 100:
        raise ValueError("n_cells en az 100 olmalidir.")
    if args.seed < 0:
        raise ValueError("seed negatif olamaz.")
    if args.uncertainty_replicates < 1:
        raise ValueError("uncertainty_replicates en az 1 olmalidir.")
    return np.linspace(args.dose_min, args.dose_max, args.dose_points)


def create_plots(
    doses: np.ndarray,
    aec_acute: np.ndarray,
    sd_acute: np.ndarray,
    aec_adaptive: np.ndarray,
    atp_fraction: np.ndarray,
    adp_fraction: np.ndarray,
    amp_fraction: np.ndarray,
    output_dir: Path,
    n_cells: int,
    seed: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    ax = axes[0]
    ax.plot(doses, aec_acute, "o-", color="#1f4e78", lw=2.2, ms=4,
            label="Tekil Akut Doz ($D$)")
    ax.fill_between(
        doses,
        np.clip(aec_acute - sd_acute, 0.0, 1.0),
        np.clip(aec_acute + sd_acute, 0.0, 1.0),
        color="#1f4e78",
        alpha=0.18,
        label="$\\pm$ 1 Monte Carlo guven araligi",
    )
    ax.plot(doses, aec_adaptive, "s--", color="#2e7d32", lw=2.0, ms=4,
            label="On Uyarim + $D_2$")
    ax.axhspan(0.75, 0.95, color="#a5d6a7", alpha=0.25,
               label="Fizyolojik aralik (0.75-0.95)")
    ax.axhline(0.70, color="#ef6c00", ls=":", lw=1.6,
               label="AMPK aktivasyon eşiği (0.70)")
    ax.axhline(0.50, color="#c62828", ls="-.", lw=1.6,
               label="Apoptoz/Nekroz eşiği (0.50)")
    ax.set(
        title="A) Doz-Bagimli Adenilat Enerji Yukune (AEC)",
        xlabel="Iyonlastirici radyasyon dozu, $D$ (Gy)",
        ylabel="AEC",
        ylim=(0.25, 1.02),
    )
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="lower left", fontsize=7.8, framealpha=0.9)

    ax2 = axes[1]
    dose_cases = [
        (0.0, "#1f4e78", "0.0 Gy (Kontrol)"),
        (1.2, "#ef6c00", "1.2 Gy (AMPK Stresi)"),
        (3.0, "#c62828", "3.0 Gy (Enerji Cokusu)"),
    ]
    t_eval = np.linspace(0.0, PARAMETERS.simulation_end, 600)
    for dose_case, color, label in dose_cases:
        stress_mean, _, _ = monte_carlo_damage_fraction(
            dose_case, n_cells=n_cells, seed=seed
        )
        solution = solve_adenylate_kinetics(stress_mean, t_eval)
        atp = np.maximum(solution.y[0], 0.0)
        adp = np.maximum(solution.y[1], 0.0)
        amp = np.maximum(solution.y[2], 0.0)
        aec_time = (atp + 0.5 * adp) / (atp + adp + amp)
        ax2.plot(t_eval, aec_time, color=color, lw=2.0, label=label)

    ax2.axhline(0.70, color="#ef6c00", ls=":", lw=1.2)
    ax2.axhline(0.50, color="#c62828", ls="-.", lw=1.2)
    ax2.set(
        title="B) Zamana Bagli AEC Kinetik Evrimi",
        xlabel="Zaman, $t$ (saniye)",
        ylabel="Anlik AEC(t)",
        ylim=(0.25, 1.02),
    )
    ax2.grid(True, ls="--", alpha=0.4)
    ax2.legend(loc="lower left", fontsize=8.5)

    ax3 = axes[2]
    ax3.plot(doses, atp_fraction, "-", color="#1565c0", lw=2.2, label="ATP Mol Kesri")
    ax3.plot(doses, adp_fraction, "--", color="#f9a825", lw=2.2, label="ADP Mol Kesri")
    ax3.plot(doses, amp_fraction, "-.", color="#ad1457", lw=2.2, label="AMP Mol Kesri")
    ax3.set(
        title="C) Doza Bagli Adenilat Dagilimi",
        xlabel="Iyonlastirici radyasyon dozu (Gy)",
        ylabel="Mol Kesri ([X] / Toplam Adenilat)",
        ylim=(0.0, 1.0),
    )
    ax3.grid(True, ls="--", alpha=0.4)
    ax3.legend(loc="center right", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(output_dir / "doz_bagimli_aec_egrileri.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


class SimulationGui:
    def __init__(self, root: tk.Tk, default_args: argparse.Namespace) -> None:
        self.root = root
        self.default_args = default_args
        self.event_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.running = False

        root.title("Doz Bagimli AEC Simulasyonu")
        root.geometry("1450x850")
        root.minsize(1100, 650)

        main = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        right = ttk.Frame(main, width=390)
        main.add(left, weight=3)
        main.add(right, weight=1)

        self.figure = Figure(figsize=(12, 7.5), dpi=100)
        self.figure.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.12, wspace=0.28)
        self.canvas = FigureCanvasTkAgg(self.figure, master=left)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.axes = self.figure.subplots(1, 3)

        controls = ttk.LabelFrame(right, text="Simulasyon Ayarlari")
        controls.pack(fill=tk.X, padx=8, pady=8)

        self.entries: dict[str, tk.StringVar] = {}
        fields = [
            ("dose_min", "Minimum doz (Gy)", str(default_args.dose_min)),
            ("dose_max", "Maksimum doz (Gy)", str(default_args.dose_max)),
            ("dose_points", "Doz noktasi", str(default_args.dose_points)),
            ("priming_dose", "On uyarim D1 (Gy)", str(default_args.priming_dose)),
            ("n_cells", "Hucre sayisi", str(default_args.n_cells)),
            ("seed", "Rastgele tohum", str(default_args.seed)),
            ("replicates", "Tekrar sayisi", str(default_args.uncertainty_replicates)),
        ]
        for row, (key, label, value) in enumerate(fields):
            ttk.Label(controls, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=4)
            variable = tk.StringVar(value=value)
            self.entries[key] = variable
            ttk.Entry(controls, textvariable=variable, width=16).grid(
                row=row, column=1, sticky="e", padx=8, pady=4
            )

        self.run_button = ttk.Button(
            controls, text="Simulasyonu Calistir", command=self.start_simulation
        )
        self.run_button.grid(row=len(fields), column=0, columnspan=2, sticky="ew", padx=8, pady=8)

        status_frame = ttk.LabelFrame(right, text="Durum")
        status_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.status_var = tk.StringVar(value="Hazir")
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor="w", padx=8, pady=4)
        self.progress = ttk.Progressbar(status_frame, maximum=100)
        self.progress.pack(fill=tk.X, padx=8, pady=(0, 8))

        summary_frame = ttk.LabelFrame(right, text="Son Olcumler")
        summary_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.summary_vars: dict[str, tk.StringVar] = {}
        for key, label in [
            ("max_dose", "Maks. doz"),
            ("aec_min", "Minimum AEC"),
            ("aec_at_max", "Maks. doz AEC"),
            ("elapsed", "Sure"),
        ]:
            row = ttk.Frame(summary_frame)
            row.pack(fill=tk.X, padx=8, pady=2)
            ttk.Label(row, text=label + ":").pack(side=tk.LEFT)
            variable = tk.StringVar(value="-")
            self.summary_vars[key] = variable
            ttk.Label(row, textvariable=variable).pack(side=tk.RIGHT)

        table_frame = ttk.LabelFrame(right, text="Doz Sonuclari")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        columns = ("dose", "aec", "sd", "atp", "amp")
        self.table = ttk.Treeview(
            table_frame, columns=columns, show="headings", height=18
        )
        headings = ("Gy", "AEC", "AEC SD", "ATP", "AMP")
        widths = (48, 62, 62, 62, 62)
        for column, heading, width in zip(columns, headings, widths):
            self.table.heading(column, text=heading)
            self.table.column(column, width=width, anchor="center")
        y_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.table.yview)
        self.table.configure(yscrollcommand=y_scroll.set)
        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.root.after(100, self.process_queue)

    def start_simulation(self) -> None:
        if self.running:
            return
        try:
            dose_min = validate_dose(float(self.entries["dose_min"].get()), "dose_min")
            dose_max = validate_dose(float(self.entries["dose_max"].get()), "dose_max")
            dose_points = int(self.entries["dose_points"].get())
            priming_dose = validate_dose(
                float(self.entries["priming_dose"].get()), "priming_dose"
            )
            n_cells = int(self.entries["n_cells"].get())
            seed = int(self.entries["seed"].get())
            replicates = int(self.entries["replicates"].get())
            if dose_max < dose_min:
                raise ValueError("Maksimum doz, minimum dozdan buyuk olmalidir.")
            if dose_points < 2 or n_cells < 100 or replicates < 1 or seed < 0:
                raise ValueError("Doz noktasi, hucre sayisi, tekrar ve tohum degerlerini kontrol edin.")
        except (TypeError, ValueError) as exc:
            self.status_var.set(str(exc))
            return

        doses = np.linspace(dose_min, dose_max, dose_points)
        self.running = True
        self.run_button.state(["disabled"])
        self.progress["value"] = 0
        self.status_var.set("Simulasyon baslatildi...")
        for item in self.table.get_children():
            self.table.delete(item)

        def progress(dose_index: int, dose: float, replicate: int, offset: int = 0) -> None:
            denominator = 2 * len(doses) * replicates
            completed = offset + dose_index * replicates + replicate
            percentage = 100.0 * completed / max(denominator, 1)
            self.event_queue.put(
                ("progress", (percentage, f"{dose:.3f} Gy - tekrar {replicate}/{replicates}"))
            )

        def worker() -> None:
            import time
            started = time.perf_counter()
            try:
                output_dir = Path(self.default_args.output_dir)
                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                except (OSError, UnicodeError):
                    output_dir = Path(r"C:\AEC_CIKTILAR")
                    output_dir.mkdir(parents=True, exist_ok=True)
                acute, acute_sd, atp, adp, amp = simulate_dose_response(
                    doses,
                    priming_dose=0.0,
                    n_cells=n_cells,
                    seed=seed,
                    uncertainty_replicates=replicates,
                    progress_callback=progress,
                )
                adaptive, adaptive_sd, _, _, _ = simulate_dose_response(
                    doses,
                    priming_dose=priming_dose,
                    n_cells=n_cells,
                    seed=seed,
                    uncertainty_replicates=replicates,
                    progress_callback=lambda i, d, r: progress(
                        i, d, r, offset=len(doses) * replicates
                    ),
                )
                results = pd.DataFrame({
                    "Doz_Gy": doses,
                    "AEC_Akut": np.round(acute, 4),
                    "AEC_Akut_SD": np.round(acute_sd, 4),
                    f"AEC_Adaptif_D1_{priming_dose:g}Gy": np.round(adaptive, 4),
                    f"AEC_Adaptif_D1_{priming_dose:g}Gy_SD": np.round(adaptive_sd, 4),
                    "ATP_Mol_Kesri": np.round(atp, 4),
                    "ADP_Mol_Kesri": np.round(adp, 4),
                    "AMP_Mol_Kesri": np.round(amp, 4),
                })
                csv_path = output_dir / "simulasyon_ciktilari.csv"
                results.to_csv(csv_path, index=False)
                elapsed = time.perf_counter() - started
                self.event_queue.put(
                    (
                        "done",
                        {
                            "doses": doses,
                            "acute": acute,
                            "sd": acute_sd,
                            "adaptive": adaptive,
                            "atp": atp,
                            "adp": adp,
                            "amp": amp,
                            "elapsed": elapsed,
                            "csv": csv_path,
                            "n_cells": n_cells,
                            "seed": seed,
                            "output_dir": output_dir,
                        },
                    )
                )
            except Exception as exc:
                self.event_queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def process_queue(self) -> None:
        try:
            while True:
                kind, payload = self.event_queue.get_nowait()
                if kind == "progress":
                    percentage, text = payload
                    self.progress["value"] = percentage
                    self.status_var.set(text)
                elif kind == "done":
                    self.show_results(payload)
                elif kind == "error":
                    self.running = False
                    self.run_button.state(["!disabled"])
                    self.status_var.set(f"Hata: {payload}")
        except queue.Empty:
            pass
        self.root.after(100, self.process_queue)

    def show_results(self, payload: dict[str, Any]) -> None:
        doses = payload["doses"]
        acute = payload["acute"]
        sd = payload["sd"]
        adaptive = payload["adaptive"]
        atp = payload["atp"]
        adp = payload["adp"]
        amp = payload["amp"]

        for item in self.table.get_children():
            self.table.delete(item)
        for index, dose in enumerate(doses):
            self.table.insert(
                "",
                tk.END,
                values=(f"{dose:.2f}", f"{acute[index]:.4f}", f"{sd[index]:.4f}",
                        f"{atp[index]:.4f}", f"{amp[index]:.4f}"),
            )

        self.summary_vars["max_dose"].set(f"{doses[-1]:.2f} Gy")
        self.summary_vars["aec_min"].set(f"{np.min(acute):.4f}")
        self.summary_vars["aec_at_max"].set(f"{acute[-1]:.4f}")
        self.summary_vars["elapsed"].set(f"{payload['elapsed']:.1f} s")

        for axis in self.axes:
            axis.clear()

        ax = self.axes[0]
        ax.plot(doses, acute, "o-", color="#1f4e78", lw=2.0, ms=4,
                label="Tekil akut doz")
        ax.fill_between(
            doses,
            np.clip(acute - sd, 0.0, 1.0),
            np.clip(acute + sd, 0.0, 1.0),
            color="#1f4e78",
            alpha=0.18,
        )
        ax.plot(doses, adaptive, "s--", color="#2e7d32", lw=2.0, ms=4,
                label="Adaptif yanit")
        ax.axhspan(0.75, 0.95, color="#a5d6a7", alpha=0.25)
        ax.axhline(0.70, color="#ef6c00", ls=":", lw=1.4)
        ax.axhline(0.50, color="#c62828", ls="-.", lw=1.4)
        ax.set(title="A) Doz-Bagimli AEC", xlabel="Doz (Gy)", ylabel="AEC", ylim=(0.25, 1.02))
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(loc="lower left", fontsize=8)

        ax2 = self.axes[1]
        t_eval = np.linspace(0.0, PARAMETERS.simulation_end, 600)
        cases = [(0.0, "#1f4e78", "0.0 Gy"), (1.2, "#ef6c00", "1.2 Gy"), (3.0, "#c62828", "3.0 Gy")]
        for dose, color, label in cases:
            stress, _, _ = monte_carlo_damage_fraction(
                dose, n_cells=payload["n_cells"], seed=payload["seed"]
            )
            solution = solve_adenylate_kinetics(stress, t_eval)
            atp_t = np.maximum(solution.y[0], 0.0)
            adp_t = np.maximum(solution.y[1], 0.0)
            amp_t = np.maximum(solution.y[2], 0.0)
            aec_t = (atp_t + 0.5 * adp_t) / (atp_t + adp_t + amp_t)
            ax2.plot(t_eval, aec_t, color=color, lw=2.0, label=label)
        ax2.axhline(0.70, color="#ef6c00", ls=":", lw=1.2)
        ax2.axhline(0.50, color="#c62828", ls="-.", lw=1.2)
        ax2.set(title="B) Zamana Bagli AEC", xlabel="Zaman (saniye)", ylabel="AEC",
                ylim=(0.25, 1.02))
        ax2.grid(True, ls="--", alpha=0.4)
        ax2.legend(loc="lower left", fontsize=8)

        ax3 = self.axes[2]
        ax3.plot(doses, atp, "-", color="#1565c0", lw=2.0, label="ATP")
        ax3.plot(doses, adp, "--", color="#f9a825", lw=2.0, label="ADP")
        ax3.plot(doses, amp, "-.", color="#ad1457", lw=2.0, label="AMP")
        ax3.set(title="C) Adenilat Dagilimi", xlabel="Doz (Gy)", ylabel="Mol Kesri",
                ylim=(0.0, 1.0))
        ax3.grid(True, ls="--", alpha=0.4)
        ax3.legend(loc="center right", fontsize=8)

        self.figure.tight_layout()
        self.canvas.draw()
        image_path = payload["output_dir"] / "doz_bagimli_aec_egrileri.png"
        try:
            image_path_string = os.fsdecode(os.path.abspath(image_path))
            self.figure.savefig(
                image_path_string,
                dpi=300,
                bbox_inches="tight",
            )
            output_message = f"Tamamlandi - CSV: {payload['csv'].name}"
        except (OSError, UnicodeError) as exc:
            output_message = f"Grafik kaydedilemedi: {exc}"
        self.progress["value"] = 100
        self.status_var.set(output_message)
        self.running = False
        self.run_button.state(["!disabled"])


def run_gui(args: argparse.Namespace) -> None:
    root = tk.Tk()
    app = SimulationGui(root, args)
    app.start_simulation()
    root.mainloop()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.headless and not args.no_show:
        run_gui(args)
        return

    try:
        doses = validate_cli_args(args)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        aec_acute, sd_acute, atp, adp, amp = simulate_dose_response(
            doses,
            priming_dose=0.0,
            n_cells=args.n_cells,
            seed=args.seed,
            uncertainty_replicates=args.uncertainty_replicates,
        )
        aec_adaptive, sd_adaptive, _, _, _ = simulate_dose_response(
            doses,
            priming_dose=args.priming_dose,
            n_cells=args.n_cells,
            seed=args.seed,
            uncertainty_replicates=args.uncertainty_replicates,
        )

        results = pd.DataFrame(
            {
                "Doz_Gy": doses,
                "AEC_Akut": np.round(aec_acute, 4),
                "AEC_Akut_SD": np.round(sd_acute, 4),
                f"AEC_Adaptif_D1_{args.priming_dose:g}Gy": np.round(aec_adaptive, 4),
                f"AEC_Adaptif_D1_{args.priming_dose:g}Gy_SD": np.round(sd_adaptive, 4),
                "ATP_Mol_Kesri": np.round(atp, 4),
                "ADP_Mol_Kesri": np.round(adp, 4),
                "AMP_Mol_Kesri": np.round(amp, 4),
            }
        )
        csv_path = args.output_dir / "simulasyon_ciktilari.csv"
        results.to_csv(csv_path, index=False)

        create_plots(
            doses=doses,
            aec_acute=aec_acute,
            sd_acute=sd_acute,
            aec_adaptive=aec_adaptive,
            atp_fraction=atp,
            adp_fraction=adp,
            amp_fraction=amp,
            output_dir=args.output_dir,
            n_cells=args.n_cells,
            seed=args.seed,
        )

        print(results.head(10).to_string(index=False))
        print(f"\nCSV kaydedildi: {csv_path}")
        print(f"Grafik kaydedildi: {args.output_dir / 'doz_bagimli_aec_egrileri.png'}")

        if not args.no_show:
            image = plt.imread(args.output_dir / "doz_bagimli_aec_egrileri.png")
            fig = plt.figure(figsize=(12, 7))
            plt.imshow(image)
            plt.axis("off")
            plt.show()
            plt.close(fig)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Hata: {exc}\n")


if __name__ == "__main__":
    main()
