"""Modern Gradio web dashboard for Steel Surface Defect Inspector.

Provides a unified navy-blue industrial quality-control interface where users can either
upload their own steel surface images or click any of the 10 pre-loaded sample scenarios
to immediately view bounding-box detections, live KPI summary cards, a structured defect
table, and an AI/rule-based quality inspection report.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Disable Gradio background telemetry so startup never hangs on restricted networks
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
from PIL import Image

from src.inference import CLASS_COLORS, DefectDetector, load_config
from src.report import DEFECT_KNOWLEDGE_TR, generate_quality_report

CONFIG_PATH = Path("configs/config.yaml")
CFG = load_config(CONFIG_PATH)
DEFAULT_CONF = float(CFG.get("model", {}).get("default_conf_threshold", 0.25))

DETECTOR = DefectDetector(config_path=CONFIG_PATH)


def warmup_preset_examples() -> None:
    """Pre-compute and cache detections for all 10 sample images at startup so switching is instant."""
    examples_dir = Path("assets/examples")
    if not examples_dir.exists():
        return
    for p in sorted(examples_dir.iterdir()):
        if p.suffix.lower() in (".jpg", ".jpeg", ".bmp", ".png"):
            try:
                pil_img = Image.open(p).convert("RGB")
                DETECTOR.predict(pil_img, conf_threshold=DEFAULT_CONF)
            except Exception:
                pass

SAMPLE_DESCRIPTIONS: Dict[str, str] = {
    "01_vertical_scratch.jpg": "Senaryo 1 - Tekil Dikey Yüzey Çiziği (1x Scratches | İnceleme)",
    "02_multi_scratches.jpg": "Senaryo 2 - Çoklu Paralel Çizikler (3x Scratches | Red)",
    "03_horizontal_scratches.jpg": "Senaryo 3 - Karma Kusur: Çizik + Çukurlaşma (Scratches + Pitted Surface)",
    "04_surface_patch_large.jpg": "Senaryo 4 - Yüksek Güvenli Yüzey Lekesi (1x Patches | %89.7 Güven)",
    "05_multi_patches.jpg": "Senaryo 5 - Çoklu Bölgesel Oksidasyon Lekesi (3x Patches | Red)",
    "06_slag_inclusion.jpg": "Senaryo 6 - Belirgin Cüruf Kalıntısı (2x Inclusion | %88.4 Güven)",
    "07_pitted_surface.jpg": "Senaryo 7 - Düşük Güvenli Sınır Anomalisi (1x Inclusion | %37.1 Güven)",
    "08_rolled_in_scale.jpg": "Senaryo 8 - Ağır Yüzey Kusuru Yoğunluğu (10x Tespit | Kritik Red)",
    "09_crazing_network.jpg": "Senaryo 9 - Çoklu Sınıf: Yama + Dikey Çizik (Patches + Scratches)",
    "10_clean_reference.jpg": "Senaryo 10 - Kusursuz Referans Çelik Sac (0 Kusur | ONAY / PASS)",
}

SAMPLE_DETAILS: Dict[str, str] = {
    "Senaryo 1 - Tekil Dikey Yüzey Çiziği (1x Scratches | İnceleme)": (
        "Hadde yönünde uzanan tekil doğrusal çizik anomalisi. Ortalama güven orta seviyede olduğu için operatör incelemesi önerilir."
    ),
    "Senaryo 2 - Çoklu Paralel Çizikler (3x Scratches | Red)": (
        "Taşıyıcı rulo veya kılavuz teması kaynaklı 3 adet paralel çizik tespiti. Çoklu kusur nedeniyle kalite ön kararı: RED."
    ),
    "Senaryo 3 - Karma Kusur: Çizik + Çukurlaşma (Scratches + Pitted Surface)": (
        "Aynı sac yüzeyinde hem yatay mekanik çizikler (scratches) hem de noktasal yüzey çukurlaşması (pitted_surface) bir arada tespit edilmiştir."
    ),
    "Senaryo 4 - Yüksek Güvenli Yüzey Lekesi (1x Patches | %89.7 Güven)": (
        "Merkez bölgede yüksek güven skoruyla (%89.7) saptanan geniş bölgesel oksidasyon / asitleme lekesi (patches)."
    ),
    "Senaryo 5 - Çoklu Bölgesel Oksidasyon Lekesi (3x Patches | Red)": (
        "Sac yüzeyinin farklı bölgelerine yayılmış 3 ayrı yüzey lekesi (patches). Yüzey kaplama öncesi temizlik hattı kontrol edilmelidir."
    ),
    "Senaryo 6 - Belirgin Cüruf Kalıntısı (2x Inclusion | %88.4 Güven)": (
        "Döküm/haddeleme aşamasından yüzeye hapsolmuş 2 adet yüksek güvenli (%88.4) metalik olmayan kalıntı (inclusion)."
    ),
    "Senaryo 7 - Düşük Güvenli Sınır Anomalisi (1x Inclusion | %37.1 Güven)": (
        "Varsayılan 0.25 eşiğinde yakalanan düşük-orta güvenli (%37.1) mikro anomali. Güven eşiği 0.40 üzerine çekildiğinde filtrelenir."
    ),
    "Senaryo 8 - Ağır Yüzey Kusuru Yoğunluğu (10x Tespit | Kritik Red)": (
        "Yüzey genelinde yaygın tufal/kalıntı dağılımı (10 adet tespit). Doğrudan kalite reddi gerektiren ağır kusurlu numune."
    ),
    "Senaryo 9 - Çoklu Sınıf: Yama + Dikey Çizik (Patches + Scratches)": (
        "Sol-orta bölgede belirgin yüzey lekesi (patches) ve sağ kenarda dikey çizik (scratches) içeren çoklu sınıf senaryosu."
    ),
    "Senaryo 10 - Kusursuz Referans Çelik Sac (0 Kusur | ONAY / PASS)": (
        "Herhangi bir yüzey kusuru içermeyen standart referans çelik sac dokusu. Kalite ön kararı: ONAY (PASS)."
    ),
}

CUSTOM_CSS = """
:root, .dark, body, .gradio-container {
    --color-accent: #2563eb !important;
    --color-accent-soft: rgba(37, 99, 235, 0.18) !important;
    --border-color-accent: #38bdf8 !important;
    --border-color-primary: rgba(148, 163, 184, 0.22) !important;
    --background-fill-primary: #0f172a !important;
    --background-fill-secondary: #1e293b !important;
    --block-background-fill: #0f1e36 !important;
    --block-border-color: rgba(148, 163, 184, 0.22) !important;
    --block-label-background-fill: rgba(15, 23, 42, 0.85) !important;
    --block-label-text-color: #cbd5e1 !important;
    --block-title-text-color: #e2e8f0 !important;
    --body-background-fill: #09111f !important;
    --body-text-color: #f1f5f9 !important;
    --body-text-color-subdued: #94a3b8 !important;
    --input-background-fill: #0f172a !important;
    --input-border-color: rgba(148, 163, 184, 0.28) !important;
    --table-even-background-fill: #0f172a !important;
    --table-odd-background-fill: #16233b !important;
    --table-border-color: rgba(148, 163, 184, 0.2) !important;
    --slider-color: #2563eb !important;
    --checkbox-background-color-selected: #2563eb !important;
}

.gradio-container {
    max-width: 1380px !important;
    margin: 0 auto !important;
    background: #09111f !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* Unified Navy-Blue Frame Design (matches top hero banner across the entire page) */
.hero-banner,
.navy-panel,
.kpi-card,
.report-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0f2942 100%) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(148, 163, 184, 0.24) !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 25px -5px rgba(5, 10, 24, 0.55) !important;
}

.hero-banner {
    padding: 26px 32px !important;
    margin-bottom: 18px !important;
}

.navy-panel {
    padding: 22px 24px !important;
    margin-bottom: 18px !important;
}

.hero-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: 12px;
}

.hero-title {
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0 0 6px 0;
    color: #ffffff !important;
}

.hero-subtitle {
    font-size: 0.96rem;
    color: #cbd5e1 !important;
    margin: 0;
    max-width: 860px;
    line-height: 1.55;
}

.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 16px;
}

.sys-badge {
    display: inline-flex;
    align-items: center;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    background: rgba(30, 58, 138, 0.45);
    color: #e2e8f0;
    border: 1px solid rgba(96, 165, 250, 0.32);
}

.sys-badge.active {
    background: rgba(37, 99, 235, 0.35);
    color: #93c5fd;
    border-color: rgba(56, 189, 248, 0.55);
}

/* KPI Grid with identical Navy Frame styling */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 6px 0 18px 0;
}

@media (max-width: 900px) {
    .kpi-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

.kpi-card {
    padding: 18px 20px !important;
}

.kpi-label {
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #93c5fd !important;
    margin-bottom: 6px;
}

.kpi-value {
    font-size: 1.45rem;
    font-weight: 800;
    color: #ffffff !important;
    line-height: 1.2;
}

.kpi-sub {
    font-size: 0.78rem;
    color: #cbd5e1 !important;
    margin-top: 5px;
}

/* Status colors: ONLY 'İNCELEME' uses orange; everything else uses green/red/blue */
.status-pass { color: #34d399 !important; }
.status-warn { color: #f59e0b !important; }
.status-fail { color: #f87171 !important; }

.section-header {
    font-size: 1.12rem;
    font-weight: 700;
    color: #e2e8f0 !important;
    margin: 0 0 14px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.2);
}

.section-header-centered {
    font-size: 1.2rem;
    font-weight: 700;
    color: #ffffff !important;
    text-align: center !important;
    margin: 4px 0 16px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.2);
}

/* Center the 3rd section Tab headers (Kalite Kontrol Raporu | Kusur Konum... | Yapılandırılmış JSON) */
.centered-tabs [role="tablist"],
.centered-tabs .tab-nav {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 12px !important;
    border-bottom: 1px solid rgba(148, 163, 184, 0.24) !important;
    padding-bottom: 8px !important;
    margin-bottom: 14px !important;
}

.centered-tabs [role="tab"],
.centered-tabs .tab-nav button {
    background: rgba(15, 23, 42, 0.75) !important;
    color: #cbd5e1 !important;
    border: 1px solid rgba(148, 163, 184, 0.24) !important;
    border-radius: 10px !important;
    padding: 10px 22px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    text-align: center !important;
    transition: all 0.15s ease !important;
}

.centered-tabs [role="tab"][aria-selected="true"],
.centered-tabs .tab-nav button.selected {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
    color: #ffffff !important;
    border-color: #60a5fa !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35) !important;
}

input[type="range"] {
    accent-color: #2563eb !important;
}

.report-box {
    padding: 22px 26px !important;
}

.report-box, .report-box * {
    color: #f1f5f9 !important;
}

.scenario-info-box {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(96, 165, 250, 0.35);
    border-left: 4px solid #38bdf8;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 14px;
}

.scenario-info-title {
    font-size: 0.92rem;
    font-weight: 700;
    color: #93c5fd;
    margin-bottom: 4px;
}

.scenario-info-desc {
    font-size: 0.86rem;
    color: #e2e8f0;
    line-height: 1.45;
}

.legend-bar {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 12px;
    padding: 12px 16px;
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 12px;
    margin-top: 10px;
    font-size: 0.8rem;
    color: #e2e8f0;
}

.legend-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-weight: 600;
}

.legend-dot {
    width: 11px;
    height: 11px;
    border-radius: 3px;
    display: inline-block;
}
"""

UI_CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    **CLASS_COLORS,
    "rolled-in_scale": (99, 102, 241),
}


def build_scenario_banner_html(scenario_label: str) -> str:
    """Render an active scenario explanation card in navy-blue theme."""
    if not scenario_label or scenario_label == "Özel Yüklenen Görsel":
        title = "Seçili Senaryo: Kullanıcı Tarafından Yüklenen Özel Görsel"
        detail = "Yüklediğiniz çelik yüzey görseli YOLOv11 modeli ile gerçek zamanlı olarak analiz edilmiştir."
    else:
        title = f"Seçili Senaryo: {scenario_label}"
        detail = SAMPLE_DETAILS.get(
            scenario_label,
            "Seçilen örnek çelik sac yüzeyi üzerinde tespit edilen kusurlar ve kalite raporu yukarıda güncellenmiştir.",
        )
    return f"""
    <div class="scenario-info-box">
        <div class="scenario-info-title">{title}</div>
        <div class="scenario-info-desc">{detail}</div>
    </div>
    """


def build_kpi_html(payload: Dict[str, Any]) -> str:
    """Render 4 navy-framed KPI cards summarizing the inspection result."""
    if not payload:
        return """
        <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-label">Kalite Ön Kararı</div><div class="kpi-value">—</div><div class="kpi-sub">Görsel bekleniyor</div></div>
            <div class="kpi-card"><div class="kpi-label">Toplam Kusur Sayısı</div><div class="kpi-value">0</div><div class="kpi-sub">Adet</div></div>
            <div class="kpi-card"><div class="kpi-label">Baskın Kusur Türü</div><div class="kpi-value">Yok</div><div class="kpi-sub">—</div></div>
            <div class="kpi-card"><div class="kpi-label">Ortalama Model Güveni</div><div class="kpi-value">%0.0</div><div class="kpi-sub">Eşik: 0.25</div></div>
        </div>
        """

    total = int(payload.get("total_detections", 0))
    conf_thresh = float(payload.get("confidence_threshold", DEFAULT_CONF))
    summary: Dict[str, Dict[str, Any]] = payload.get("summary_by_class", {})
    mode_used = payload.get("report_generation_mode", "rule_based")
    mode_label = "LLM Raporu" if mode_used == "llm" else "Kural Tabanlı Şablon"

    if total == 0 or not summary:
        status_text = "ONAY (PASS)"
        status_cls = "status-pass"
        status_sub = "Yüzeyde kusur saptanmadı"
        dominant_tr = "Kusur Yok"
        dominant_en = "Temiz Yüzey"
        avg_conf_pct = "—"
    else:
        dom_cls, dom_stats = max(
            summary.items(),
            key=lambda kv: (kv[1].get("count", 0), kv[1].get("max_confidence", 0.0)),
        )
        dominant_tr = DEFECT_KNOWLEDGE_TR.get(dom_cls, {}).get("label_tr", dom_cls).split(" (")[0]
        dominant_en = f"Sınıf: {dom_cls} ({dom_stats.get('count', 1)} adet)"

        all_confs = [float(d["confidence"]) for d in payload.get("detections", [])]
        mean_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0
        avg_conf_pct = f"%{mean_conf * 100:.1f}"

        if total >= 3 or mean_conf >= 0.70:
            status_text = "RED / KUSURLU"
            status_cls = "status-fail"
            status_sub = "Operatör müdahalesi gerekli"
        else:
            status_text = "İNCELEME"
            status_cls = "status-warn"
            status_sub = "Şüpheli yüzey anomalisi"

    return f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Kalite Ön Kararı</div>
            <div class="kpi-value {status_cls}">{status_text}</div>
            <div class="kpi-sub">{status_sub}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Toplam Kusur Sayısı</div>
            <div class="kpi-value">{total} Adet</div>
            <div class="kpi-sub">{len(summary)} farklı kusur sınıfı</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Baskın Kusur Türü</div>
            <div class="kpi-value" style="font-size:1.18rem;">{dominant_tr}</div>
            <div class="kpi-sub">{dominant_en}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Ortalama Model Güveni</div>
            <div class="kpi-value">{avg_conf_pct}</div>
            <div class="kpi-sub">Eşik: {conf_thresh:.2f} | {mode_label}</div>
        </div>
    </div>
    """


def build_legend_html() -> str:
    """Render a color legend for the 6 NEU-DET defect classes without any orange color."""
    items = []
    for cls_name, rgb in UI_CLASS_COLORS.items():
        tr_name = DEFECT_KNOWLEDGE_TR.get(cls_name, {}).get("label_tr", cls_name)
        hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        items.append(
            f'<span class="legend-item">'
            f'<span class="legend-dot" style="background:{hex_color};"></span>'
            f"{tr_name}</span>"
        )
    return f'<div class="legend-bar">{" ".join(items)}</div>'


def format_detections_table(detections: List[dict[str, Any]]) -> List[List[Any]]:
    """Convert raw detection dicts into rows for the Gradio Dataframe component."""
    rows: List[List[Any]] = []
    for det in detections:
        cls_name = str(det["class_name"])
        label_tr = DEFECT_KNOWLEDGE_TR.get(cls_name, {}).get("label_tr", cls_name)
        bbox_str = ", ".join(str(int(round(v))) for v in det["bbox_xyxy"])
        rows.append(
            [
                det["id"],
                label_tr,
                cls_name,
                f"%{float(det['confidence']) * 100:.1f}",
                det["region"],
                f"[{bbox_str}]",
            ]
        )
    return rows


def inspect_surface(
    image: Optional[Image.Image],
    conf_threshold: float,
) -> Tuple[Optional[Image.Image], str, List[List[Any]], str, dict[str, Any]]:
    """Run YOLO detection and generate KPI cards, table, report, and JSON for the UI."""
    if image is None:
        return (
            None,
            build_kpi_html({}),
            [],
            "Lütfen sol panelden bir çelik yüzey görseli yükleyin veya **10 Hazır Örnek Senaryodan** birini seçin.",
            {},
        )

    annotated_img, detections, payload = DETECTOR.predict(
        image=image,
        conf_threshold=conf_threshold,
    )
    table_rows = format_detections_table(detections)
    report_md, report_mode = generate_quality_report(payload, config_path=CONFIG_PATH)
    payload["report_generation_mode"] = report_mode
    kpi_html = build_kpi_html(payload)

    return annotated_img, kpi_html, table_rows, report_md, payload


def handle_custom_upload(
    image: Optional[Image.Image],
    conf_threshold: float,
) -> Tuple[Optional[Image.Image], str, List[List[Any]], str, dict[str, Any], str]:
    """Handle user-uploaded image and update the scenario explanation banner accordingly."""
    annotated_img, kpi_html, table_rows, report_md, payload = inspect_surface(
        image, conf_threshold
    )
    banner_html = build_scenario_banner_html("Özel Yüklenen Görsel")
    return annotated_img, kpi_html, table_rows, report_md, payload, banner_html


def discover_examples() -> List[List[Any]]:
    """Return up to 10 sample images from assets/examples/ with their preset thresholds."""
    examples_dir = Path("assets/examples")
    if not examples_dir.exists():
        return []
    sample_files = sorted(
        p for p in examples_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".bmp", ".png")
    )
    rows: List[List[Any]] = []
    for p in sample_files[:10]:
        desc = SAMPLE_DESCRIPTIONS.get(p.name, f"Örnek Görsel ({p.name})")
        rows.append([str(p), DEFAULT_CONF, desc])
    return rows


def load_preset_scenario(
    scenario_label: str,
    conf_threshold: float,
) -> Tuple[
    Optional[Image.Image],
    Optional[Image.Image],
    str,
    List[List[Any]],
    str,
    dict[str, Any],
    str,
]:
    """Load one of the 10 preset sample images by its label and update all outputs + explanation."""
    examples_dir = Path("assets/examples")
    target_path: Optional[Path] = None
    for fname, desc in SAMPLE_DESCRIPTIONS.items():
        if desc == scenario_label:
            candidate = examples_dir / fname
            if candidate.exists():
                target_path = candidate
                break

    if target_path is None:
        sample_files = sorted(
            p for p in examples_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".bmp", ".png")
        )
        if sample_files:
            target_path = sample_files[0]

    banner_html = build_scenario_banner_html(scenario_label)

    if target_path is None or not target_path.exists():
        empty_kpi = build_kpi_html({})
        return None, None, empty_kpi, [], "Örnek görsel bulunamadı.", {}, banner_html

    pil_img = Image.open(target_path).convert("RGB")
    annotated_img, kpi_html, table_rows, report_md, payload = inspect_surface(
        pil_img, conf_threshold
    )
    return pil_img, annotated_img, kpi_html, table_rows, report_md, payload, banner_html


def handle_example_click(
    image: Optional[Image.Image],
    conf_threshold: float,
    scenario_name: str = "",
) -> Tuple[Optional[Image.Image], str, List[List[Any]], str, dict[str, Any], str]:
    """Wrapper for gr.Examples click that updates inspection outputs and banner in a single round-trip."""
    annotated_img, kpi_html, table_rows, report_md, payload = inspect_surface(
        image, conf_threshold
    )
    banner_html = build_scenario_banner_html(scenario_name)
    return (
        annotated_img,
        kpi_html,
        table_rows,
        report_md,
        payload,
        banner_html,
    )


def build_interface() -> gr.Blocks:
    """Construct the navy-themed Gradio Blocks dashboard with no emojis."""
    has_token = bool(os.environ.get("HF_TOKEN", "").strip())
    llm_model_id = CFG.get("llm", {}).get("model_id", "Qwen/Qwen2.5-7B-Instruct")

    weights_badge = (
        '<span class="sys-badge active">Model: YOLOv11n (weights/best.pt Aktif | mAP@50: 0.735)</span>'
        if DETECTOR.using_custom_weights
        else f'<span class="sys-badge">Model: Temel Ağırlık ({DETECTOR.resolved_weights})</span>'
    )
    llm_badge = (
        f'<span class="sys-badge active">Raporlayıcı: LLM Aktif ({llm_model_id})</span>'
        if has_token
        else '<span class="sys-badge">Raporlayıcı: Kural Tabanlı Mühendislik Raporu (Çevrimdışı Mod)</span>'
    )

    examples = discover_examples()
    scenario_choices = [
        desc
        for fname, desc in SAMPLE_DESCRIPTIONS.items()
        if (Path("assets/examples") / fname).exists()
    ]
    initial_scenario = scenario_choices[0] if scenario_choices else ""

    with gr.Blocks(title="Steel Surface Defect Inspector") as demo:
        # 1. Top Navy-Blue Hero Frame + Global CSS
        gr.HTML(
            f"""
            <style>{CUSTOM_CSS}</style>
            <div class="hero-banner">
                <div class="hero-top">
                    <div>
                        <h1 class="hero-title">Steel Surface Defect Inspector</h1>
                        <p class="hero-subtitle">
                            Bursa otomotiv ve çelik yan sanayi görsel kalite kontrol süreçlerinden esinlenilmiş,
                            <strong>YOLOv11</strong> tabanlı gerçek zamanlı yüzey kusuru tespiti ve
                            <strong>Yapay Zeka Kalite Raporlama</strong> kontrol paneli.
                        </p>
                    </div>
                </div>
                <div class="badge-row">
                    {weights_badge}
                    {llm_badge}
                    <span class="sys-badge">Veri Seti: NEU-DET (6 Kusur Sınıfı)</span>
                    <span class="sys-badge">Çıkarım Hızı: ~8.5 ms / görsel</span>
                </div>
            </div>
            """
        )

        # 2. Live KPI Summary Cards (same navy frame styling)
        kpi_cards = gr.HTML(value=build_kpi_html({}))

        # 3. Main Dual-Column Inspection Workbench (wrapped in navy-panel frames)
        with gr.Row(equal_height=True):
            with gr.Column(scale=5, elem_classes=["navy-panel"]):
                gr.HTML('<div class="section-header">1. Görsel Yükle veya Hazır Senaryo Seç</div>')
                if scenario_choices:
                    scenario_dropdown = gr.Dropdown(
                        choices=scenario_choices,
                        value=initial_scenario,
                        label="10 Hazır Test Senaryosundan Hızlı Seç (Seçtiğiniz An Sonuç Üretir)",
                        interactive=True,
                    )
                else:
                    scenario_dropdown = gr.Dropdown(choices=[], label="Senaryo", visible=False)

                scenario_banner = gr.HTML(
                    value=build_scenario_banner_html(initial_scenario)
                )

                input_image = gr.Image(
                    type="pil",
                    format="jpeg",
                    label="Kendi Çelik Sac Görselinizi Yükleyin (veya Sürükleyip Bırakın)",
                    height=300,
                )

                conf_slider = gr.Slider(
                    minimum=0.05,
                    maximum=0.90,
                    value=DEFAULT_CONF,
                    step=0.05,
                    label="Güven Eşiği (Confidence Threshold)",
                    info="Görsel yüklediğinizde, senaryo seçtiğinizde veya bu çubuğu kaydırdığınızda analiz otomatik güncellenir.",
                )

            with gr.Column(scale=7, elem_classes=["navy-panel"]):
                gr.HTML('<div class="section-header">2. Kusur Tespit Haritası (Bounding Box)</div>')
                output_image = gr.Image(
                    type="pil",
                    format="jpeg",
                    label="YOLOv11 Kusur Tespiti Çıktısı",
                    height=440,
                )
                gr.HTML(build_legend_html())

        # 4. Detailed Inspection Tabs (Centered title + centered tab buttons inside navy-panel frame)
        with gr.Column(elem_classes=["navy-panel", "centered-tabs"]):
            gr.HTML(
                '<div class="section-header-centered">'
                "3. Mühendislik Kalite Raporu ve Detaylı Metrikler"
                "</div>"
            )
            with gr.Tabs():
                with gr.TabItem("Kalite Kontrol Raporu"):
                    report_markdown = gr.Markdown(
                        value="*Analiz sonucu ve kalite kontrol raporu burada görüntülenecektir.*",
                        elem_classes=["report-box"],
                    )

                with gr.TabItem("Kusur Konum ve Güven Tablosu"):
                    defect_table = gr.Dataframe(
                        headers=[
                            "#",
                            "Kusur Türü (TR)",
                            "Sınıf Kodu (EN)",
                            "Güven Skoru",
                            "Yüzey Bölgesi",
                            "Kutu Koordinatları [x1, y1, x2, y2]",
                        ],
                        datatype=["number", "str", "str", "str", "str", "str"],
                        interactive=False,
                        label="Tespit Edilen Kusurlar Tablosu",
                    )

                with gr.TabItem("Yapılandırılmış JSON (YOLO -> LLM)"):
                    raw_json = gr.JSON(label="YOLO Yapılandırılmış Çıktı")

        # 5. Pre-loaded 10 Interactive Examples Gallery (inside navy-panel frame)
        if examples:
            with gr.Column(elem_classes=["navy-panel"]):
                gr.HTML(
                    '<div class="section-header">'
                    "4. Hazır 10 Farklı Test Görseli Galerisi (Doğrudan Tıklayıp Sonuçları Görün)"
                    "</div>"
                )
                scenario_label_col = gr.Textbox(
                    visible=False,
                    label="Senaryo Adı",
                )
                gr.Examples(
                    examples=examples,
                    inputs=[input_image, conf_slider, scenario_label_col],
                    outputs=[
                        output_image,
                        kpi_cards,
                        defect_table,
                        report_markdown,
                        raw_json,
                        scenario_banner,
                    ],
                    fn=handle_example_click,
                    run_on_click=True,
                    examples_per_page=10,
                    label="10 Farklı Kusur Senaryosu (Görsele veya Satıra Tıklayın)",
                )

        # Event bindings (fully reactive)
        input_image.upload(
            fn=handle_custom_upload,
            inputs=[input_image, conf_slider],
            outputs=[
                output_image,
                kpi_cards,
                defect_table,
                report_markdown,
                raw_json,
                scenario_banner,
            ],
        )
        conf_slider.release(
            fn=inspect_surface,
            inputs=[input_image, conf_slider],
            outputs=[output_image, kpi_cards, defect_table, report_markdown, raw_json],
        )

        if scenario_choices:
            scenario_dropdown.change(
                fn=load_preset_scenario,
                inputs=[scenario_dropdown, conf_slider],
                outputs=[
                    input_image,
                    output_image,
                    kpi_cards,
                    defect_table,
                    report_markdown,
                    raw_json,
                    scenario_banner,
                ],
            )
            demo.load(
                fn=load_preset_scenario,
                inputs=[scenario_dropdown, conf_slider],
                outputs=[
                    input_image,
                    output_image,
                    kpi_cards,
                    defect_table,
                    report_markdown,
                    raw_json,
                    scenario_banner,
                ],
            )

    return demo


if __name__ == "__main__":
    warmup_preset_examples()
    app = build_interface()
    port = int(os.environ.get("PORT", 7860))
    app.launch(server_name="0.0.0.0", server_port=port, max_threads=2)
