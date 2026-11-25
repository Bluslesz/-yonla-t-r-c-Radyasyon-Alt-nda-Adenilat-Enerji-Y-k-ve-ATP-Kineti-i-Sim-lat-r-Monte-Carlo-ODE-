
from __future__ import annotations

import argparse
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.table import Table, TableStyleInfo
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "ciktilar",
    )
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


def build_results_dataframe(
    doses: np.ndarray,
    acute: np.ndarray,
    acute_sd: np.ndarray,
    adaptive: np.ndarray,
    adaptive_sd: np.ndarray,
    atp: np.ndarray,
    adp: np.ndarray,
    amp: np.ndarray,
    priming_dose: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Doz_Gy": doses,
            "AEC_Akut": np.round(acute, 4),
            "AEC_Akut_SD": np.round(acute_sd, 4),
            f"AEC_Adaptif_D1_{priming_dose:g}Gy": np.round(adaptive, 4),
            f"AEC_Adaptif_D1_{priming_dose:g}Gy_SD": np.round(adaptive_sd, 4),
            "ATP_Mol_Kesri": np.round(atp, 4),
            "ADP_Mol_Kesri": np.round(adp, 4),
            "AMP_Mol_Kesri": np.round(amp, 4),
        }
    )


def save_results(results: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    csv_path = output_dir / "simulasyon_ciktilari.csv"
    excel_path = output_dir / "simulasyon_ciktilari.xlsx"
    results.to_csv(csv_path, index=False)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        sheet_name = "AEC Sonuçları"
        results.to_excel(writer, sheet_name=sheet_name, index=False)
        worksheet = writer.sheets[sheet_name]
        worksheet.freeze_panes = "A2"
        worksheet.sheet_view.showGridLines = False

        header_fill = PatternFill(fill_type="solid", fgColor="1F4E78")
        for cell in worksheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")

        for column_index, column_name in enumerate(results.columns, start=1):
            values = results[column_name].tolist()
            display_values = [str(column_name)] + [str(value) for value in values]
            worksheet.column_dimensions[
                worksheet.cell(row=1, column=column_index).column_letter
            ].width = min(max(max(map(len, display_values)) + 2, 12), 34)

            number_format = "0.000" if column_name == "Doz_Gy" else "0.0000"
            for row_index in range(2, len(values) + 2):
                cell = worksheet.cell(row=row_index, column=column_index)
                cell.number_format = number_format
                cell.alignment = Alignment(horizontal="center")

        if not results.empty:
            table = Table(
                displayName="AECResults",
                ref=f"A1:{worksheet.cell(row=len(results) + 1, column=len(results.columns)).coordinate}",
            )
            table.tableStyleInfo = TableStyleInfo(
                name="TableStyleMedium2",
                showFirstColumn=False,
                showLastColumn=False,
                showRowStripes=True,
                showColumnStripes=False,
            )
            worksheet.add_table(table)

    return csv_path, excel_path


def style_plot_axis(axis: Any, detailed: bool = False) -> None:
    axis.set_facecolor("#FFFFFF")
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#DCE3EA", linewidth=0.8)
    axis.grid(axis="x", visible=False)
    axis.tick_params(colors="#52606D", labelsize=10 if detailed else 8)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color("#C7D0D9")
        axis.spines[spine].set_linewidth(0.8)
    axis.title.set_color("#243447")
    axis.title.set_fontweight("bold")
    axis.title.set_fontsize(12 if detailed else 10)
    axis.xaxis.label.set_color("#52606D")
    axis.yaxis.label.set_color("#52606D")
    axis.xaxis.label.set_fontsize(10 if detailed else 8)
    axis.yaxis.label.set_fontsize(10 if detailed else 8)


def annotate_threshold(axis: Any, value: float, color: str, fontsize: int = 7) -> None:
    axis.annotate(
        f"{value:.2f}",
        (1.0, value),
        xycoords=("axes fraction", "data"),
        xytext=(-4, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=color,
        fontsize=fontsize,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
    )


def plot_dose_response(
    axis: Any,
    doses: np.ndarray,
    aec_acute: np.ndarray,
    sd_acute: np.ndarray,
    aec_adaptive: np.ndarray,
    sd_adaptive: np.ndarray,
    priming_dose: float,
    detailed: bool = False,
) -> None:
    navy = "#245B84"
    teal = "#168C83"
    amber = "#D9822B"
    red = "#C45151"
    green = "#4E8B62"
    axis.plot(
        doses,
        aec_acute,
        color=navy,
        lw=2.2,
        marker="o",
        ms=5 if detailed else 4,
        markeredgecolor="white",
        markeredgewidth=0.7,
        label="Akut doz",
    )
    axis.fill_between(
        doses,
        np.clip(aec_acute - sd_acute, 0.0, 1.0),
        np.clip(aec_acute + sd_acute, 0.0, 1.0),
        color=navy,
        alpha=0.14,
        label="Akut ±1 SS",
    )
    axis.fill_between(
        doses,
        np.clip(aec_adaptive - sd_adaptive, 0.0, 1.0),
        np.clip(aec_adaptive + sd_adaptive, 0.0, 1.0),
        color=teal,
        alpha=0.12,
        label="Adaptif ±1 SS",
    )
    axis.plot(
        doses,
        aec_adaptive,
        color=teal,
        lw=2.0,
        ls="--",
        marker="s",
        ms=4.5 if detailed else 3.5,
        markeredgecolor="white",
        markeredgewidth=0.7,
        label=f"Ön uyarım ({priming_dose:g} Gy)",
    )
    axis.axhspan(
        0.75,
        0.95,
        color=green,
        alpha=0.09,
        zorder=0,
        label="Fizyolojik referans (0.75–0.95)",
    )
    axis.axhline(0.70, color=amber, ls=":", lw=1.5, label="Referans eşik 0.70")
    axis.axhline(0.50, color=red, ls="-.", lw=1.4, label="Referans eşik 0.50")
    annotate_threshold(axis, 0.70, amber, 9 if detailed else 7)
    annotate_threshold(axis, 0.50, red, 9 if detailed else 7)
    axis.set(
        title="AEC doz yanıtı",
        xlabel="Radyasyon dozu (Gy)",
        ylabel="Adenilat enerji yükü (AEC)",
        ylim=(0.0, 1.02),
    )
    if detailed:
        axis.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=9,
            frameon=False,
        )
    else:
        axis.legend(loc="upper right", fontsize=6.5, frameon=False, ncol=2)


def calculate_aec_time_courses(
    n_cells: int,
    seed: int,
) -> tuple[np.ndarray, list[tuple[str, str, np.ndarray]]]:
    time_points = np.linspace(0.0, PARAMETERS.simulation_end, 600)
    dose_cases = [
        (0.0, "#245B84", "0.0 Gy · kontrol"),
        (1.2, "#D9822B", "1.2 Gy · orta doz"),
        (3.0, "#C45151", "3.0 Gy · yüksek doz"),
    ]
    time_courses: list[tuple[str, str, np.ndarray]] = []
    for dose, color, label in dose_cases:
        stress_mean, _, _ = monte_carlo_damage_fraction(
            dose, n_cells=n_cells, seed=seed
        )
        solution = solve_adenylate_kinetics(stress_mean, time_points)
        atp = np.maximum(solution.y[0], 0.0)
        adp = np.maximum(solution.y[1], 0.0)
        amp = np.maximum(solution.y[2], 0.0)
        total = np.maximum(atp + adp + amp, np.finfo(float).tiny)
        aec_time = (atp + 0.5 * adp) / total
        time_courses.append((color, label, aec_time))
    return time_points, time_courses


def plot_aec_time_courses(
    axis: Any,
    time_points: np.ndarray,
    time_courses: Sequence[tuple[str, str, np.ndarray]],
    detailed: bool = False,
) -> None:
    amber = "#D9822B"
    red = "#C45151"
    for color, label, aec_time in time_courses:
        axis.plot(time_points, aec_time, color=color, lw=2.4 if detailed else 2.0)
        axis.annotate(
            label,
            (time_points[-1], aec_time[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=color,
            fontsize=9 if detailed else 7,
            clip_on=False,
        )

    axis.axhline(0.70, color=amber, ls=":", lw=1.5)
    axis.axhline(0.50, color=red, ls="-.", lw=1.4)
    annotate_threshold(axis, 0.70, amber, 9 if detailed else 7)
    annotate_threshold(axis, 0.50, red, 9 if detailed else 7)
    axis.set(
        title="AEC'nin zamana bağlı değişimi",
        xlabel="Zaman (s)",
        ylabel="Anlık AEC",
        ylim=(0.0, 1.02),
        xlim=(time_points[0], time_points[-1] * 1.18),
    )


def plot_adenylate_fractions(
    axis: Any,
    doses: np.ndarray,
    atp_fraction: np.ndarray,
    adp_fraction: np.ndarray,
    amp_fraction: np.ndarray,
    detailed: bool = False,
) -> None:
    axis.plot(doses, atp_fraction, color="#245B84", lw=2.2, label="ATP")
    axis.plot(doses, adp_fraction, color="#D9822B", lw=2.0, ls="--", label="ADP")
    axis.plot(doses, amp_fraction, color="#8256A5", lw=2.0, ls="-.", label="AMP")
    axis.set(
        title="Adenilat havuzlarının dağılımı",
        xlabel="Radyasyon dozu (Gy)",
        ylabel="Mol kesri",
        ylim=(0.0, 1.0),
    )
    if detailed:
        axis.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=9,
            frameon=False,
        )
    else:
        axis.legend(loc="best", fontsize=9, frameon=False)


def plot_simulation_results(
    figure: Figure,
    axes: Sequence[Any],
    doses: np.ndarray,
    aec_acute: np.ndarray,
    sd_acute: np.ndarray,
    aec_adaptive: np.ndarray,
    sd_adaptive: np.ndarray,
    atp_fraction: np.ndarray,
    adp_fraction: np.ndarray,
    amp_fraction: np.ndarray,
    time_points: np.ndarray,
    time_courses: Sequence[tuple[str, str, np.ndarray]],
    priming_dose: float,
) -> None:
    figure.set_facecolor("#FFFFFF")
    figure.suptitle(
        "Doza Bağlı AEC ve Adenilat Dinamikleri",
        fontsize=15,
        fontweight="bold",
        color="#243447",
    )
    plot_dose_response(
        axes[0], doses, aec_acute, sd_acute, aec_adaptive, sd_adaptive, priming_dose
    )
    plot_aec_time_courses(axes[1], time_points, time_courses)
    plot_adenylate_fractions(axes[2], doses, atp_fraction, adp_fraction, amp_fraction)
    for axis in axes:
        style_plot_axis(axis)
    figure.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.14, wspace=0.34)


def save_detailed_plots(
    output_dir: Path,
    doses: np.ndarray,
    aec_acute: np.ndarray,
    sd_acute: np.ndarray,
    aec_adaptive: np.ndarray,
    sd_adaptive: np.ndarray,
    atp_fraction: np.ndarray,
    adp_fraction: np.ndarray,
    amp_fraction: np.ndarray,
    time_points: np.ndarray,
    time_courses: Sequence[tuple[str, str, np.ndarray]],
    priming_dose: float,
) -> None:
    detailed_plots: list[tuple[str, str, str, Callable[[Any], None]]] = [
        (
            "aec_doz_yaniti.png",
            "A | Doza bağlı adenilat enerji yükü",
            "AEC, ATP ağırlıklı enerji durumunu 0–1 arasında özetler. "
            "Gölgeli alanlar tekrarlar arası ±1 standart sapmayı; yatay çizgiler "
            "modelde kullanılan referans eşikleri gösterir. Düşük AEC değerleri "
            "grafikte kırpılmadan gösterilir.",
            lambda axis: plot_dose_response(
                axis,
                doses,
                aec_acute,
                sd_acute,
                aec_adaptive,
                sd_adaptive,
                priming_dose,
                detailed=True,
            ),
        ),
        (
            "aec_zaman_kinetigi.png",
            "B | Zamana bağlı AEC kinetiği",
            "Eğriler kontrol, orta ve yüksek doz için adenilat ODE modelinden "
            "hesaplanmıştır. Yatay eksen simülasyon süresini, düşey eksen anlık "
            "AEC'yi gösterir. 0.70 ve 0.50 çizgileri model referans eşikleridir.",
            lambda axis: plot_aec_time_courses(
                axis, time_points, time_courses, detailed=True
            ),
        ),
        (
            "adenilat_mol_kesirleri.png",
            "C | Doza bağlı ATP, ADP ve AMP dağılımı",
            "Mol kesirleri her dozda toplam adenilat havuzuna göre normalize edilir; "
            "bu nedenle ATP, ADP ve AMP kesirlerinin toplamı yaklaşık 1'dir.",
            lambda axis: plot_adenylate_fractions(
                axis,
                doses,
                atp_fraction,
                adp_fraction,
                amp_fraction,
                detailed=True,
            ),
        ),
    ]

    for filename, title, explanation, draw_plot in detailed_plots:
        figure, axis = plt.subplots(figsize=(11, 6.4), dpi=300)
        figure.set_facecolor("#FFFFFF")
        draw_plot(axis)
        style_plot_axis(axis, detailed=True)
        figure.suptitle(title, fontsize=15, fontweight="bold", color="#243447")
        figure.text(
            0.10,
            0.035,
            explanation,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#52606D",
            wrap=True,
        )
        figure.subplots_adjust(left=0.11, right=0.76, top=0.86, bottom=0.25)
        figure.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(figure)


def create_plots(
    doses: np.ndarray,
    aec_acute: np.ndarray,
    sd_acute: np.ndarray,
    aec_adaptive: np.ndarray,
    sd_adaptive: np.ndarray,
    atp_fraction: np.ndarray,
    adp_fraction: np.ndarray,
    amp_fraction: np.ndarray,
    output_dir: Path,
    n_cells: int,
    seed: int,
    priming_dose: float,
) -> None:
    time_points, time_courses = calculate_aec_time_courses(n_cells, seed)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), dpi=300)
    plot_simulation_results(
        fig,
        axes,
        doses,
        aec_acute,
        sd_acute,
        aec_adaptive,
        sd_adaptive,
        atp_fraction,
        adp_fraction,
        amp_fraction,
        time_points,
        time_courses,
        priming_dose,
    )
    fig.savefig(output_dir / "doz_bagimli_aec_egrileri.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    save_detailed_plots(
        output_dir,
        doses,
        aec_acute,
        sd_acute,
        aec_adaptive,
        sd_adaptive,
        atp_fraction,
        adp_fraction,
        amp_fraction,
        time_points,
        time_courses,
        priming_dose,
    )


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
        ttk.Label(
            status_frame,
            textvariable=self.status_var,
            wraplength=350,
            justify="left",
        ).pack(anchor="w", padx=8, pady=4)
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
                output_dir = Path(self.default_args.output_dir).expanduser().resolve()
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
                results = build_results_dataframe(
                    doses,
                    acute,
                    acute_sd,
                    adaptive,
                    adaptive_sd,
                    atp,
                    adp,
                    amp,
                    priming_dose,
                )
                csv_path, excel_path = save_results(results, output_dir)
                elapsed = time.perf_counter() - started
                self.event_queue.put(
                    (
                        "done",
                        {
                            "doses": doses,
                            "acute": acute,
                            "sd": acute_sd,
                            "adaptive": adaptive,
                            "adaptive_sd": adaptive_sd,
                            "atp": atp,
                            "adp": adp,
                            "amp": amp,
                            "elapsed": elapsed,
                            "csv": csv_path,
                            "excel": excel_path,
                            "n_cells": n_cells,
                            "seed": seed,
                            "priming_dose": priming_dose,
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
        adaptive_sd = payload["adaptive_sd"]
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
        time_points, time_courses = calculate_aec_time_courses(
            payload["n_cells"], payload["seed"]
        )
        plot_simulation_results(
            self.figure,
            self.axes,
            doses,
            acute,
            sd,
            adaptive,
            adaptive_sd,
            atp,
            adp,
            amp,
            time_points,
            time_courses,
            payload["priming_dose"],
        )
        self.canvas.draw()
        image_path = payload["output_dir"] / "doz_bagimli_aec_egrileri.png"
        try:
            image_path_string = os.fsdecode(os.path.abspath(image_path))
            self.figure.savefig(
                image_path_string,
                dpi=300,
                bbox_inches="tight",
            )
            save_detailed_plots(
                payload["output_dir"],
                doses,
                acute,
                sd,
                adaptive,
                adaptive_sd,
                atp,
                adp,
                amp,
                time_points,
                time_courses,
                payload["priming_dose"],
            )
            output_message = (
                "Çıktılar kaydedildi:\n"
                f"{payload['output_dir']}"
            )
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
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

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

        results = build_results_dataframe(
            doses,
            aec_acute,
            sd_acute,
            aec_adaptive,
            sd_adaptive,
            atp,
            adp,
            amp,
            args.priming_dose,
        )
        csv_path, excel_path = save_results(results, output_dir)

        create_plots(
            doses=doses,
            aec_acute=aec_acute,
            sd_acute=sd_acute,
            aec_adaptive=aec_adaptive,
            sd_adaptive=sd_adaptive,
            atp_fraction=atp,
            adp_fraction=adp,
            amp_fraction=amp,
            output_dir=output_dir,
            n_cells=args.n_cells,
            seed=args.seed,
            priming_dose=args.priming_dose,
        )

        print(results.head(10).to_string(index=False))
        print(f"\nCSV kaydedildi: {csv_path}")
        print(f"Excel kaydedildi: {excel_path}")
        print(f"Grafik kaydedildi: {output_dir / 'doz_bagimli_aec_egrileri.png'}")
        print(f"A grafiği kaydedildi: {output_dir / 'aec_doz_yaniti.png'}")
        print(f"B grafiği kaydedildi: {output_dir / 'aec_zaman_kinetigi.png'}")
        print(f"C grafiği kaydedildi: {output_dir / 'adenilat_mol_kesirleri.png'}")

        if not args.no_show:
            image = plt.imread(output_dir / "doz_bagimli_aec_egrileri.png")
            fig = plt.figure(figsize=(12, 7))
            plt.imshow(image)
            plt.axis("off")
            plt.show()
            plt.close(fig)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Hata: {exc}\n")


if __name__ == "__main__":
    main()
