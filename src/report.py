"""Quality inspection report generator using Hugging Face Inference API with rule-based fallback.

Reads the LLM model identifier from configs/config.yaml and the API token exclusively
from the HF_TOKEN environment variable. Never hardcodes credentials. If HF_TOKEN is absent
or the API call fails, automatically generates a deterministic template-based report.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

from src.inference import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Domain knowledge mapping for cautious root-cause hypotheses and actions (Turkish)
DEFECT_KNOWLEDGE_TR: Dict[str, Dict[str, str]] = {
    "crazing": {
        "label_tr": "Kılcal Çatlak (Crazing)",
        "cause": "Haddeleme veya soğutma esnasındaki ısıl gerilmeler ya da kalıp-yüzey sürtünmesi kaynaklı olabilir.",
        "action": "Soğutma hızı profillerini ve haddeleme sıcaklık dengesini kontrol edin; çatlak derinliğini doğrulayın.",
    },
    "inclusion": {
        "label_tr": "Kalıntı / Cüruf Batması (Inclusion)",
        "cause": "Sıvı çelik dökümü sırasında metalik olmayan oksit/cüruf kalıntılarının yüzeye yakın hapsolmasından kaynaklanıyor olabilir.",
        "action": "Döküm potası cüruf temizliğini, refrakter aşınmasını ve tufal yıkama basıncını gözden geçirin.",
    },
    "patches": {
        "label_tr": "Yüzey Lekesi / Yama (Patches)",
        "cause": "Asitleme (pickling) banyosu kalıntıları, düzensiz yağlama veya yerel oksidasyon kaynaklı olabilir.",
        "action": "Yüzey temizleme/durulama hattını ve koruyucu yağlama nozullarını denetleyin.",
    },
    "pitted_surface": {
        "label_tr": "Çukurlaşma / Oyuk Yüzey (Pitted Surface)",
        "cause": "Merda (iş merdanesi) yüzeyindeki yerel aşınmalar veya kopan sert tufal parçacıklarının sac yüzeyine batmasıyla oluşmuş olabilir.",
        "action": "İş merdanelerinin yüzey durumunu kontrol edin ve yüksek basınçlı suyla tufal kırma (descaling) hattını doğrulayın.",
    },
    "rolled-in_scale": {
        "label_tr": "Hadde Tufalı Batması (Rolled-in Scale)",
        "cause": "Sıcak haddeleme öncesinde yüzeydeki oksit tabakasının (tufalın) tam temizlenemeyip merdaneler altında saca basılmasından kaynaklanabilir.",
        "action": "Tufal yıkama (descaler) nozul basınçlarını, açılarını ve fırın atmosferindeki oksidasyon seviyesini inceleyin.",
    },
    "scratches": {
        "label_tr": "Yüzey Çiziği (Scratches)",
        "cause": "Bant taşıma ruloları, kılavuz yan duvarları veya sarıcı-açıcı mekanizmalardaki mekanik sürtünme ve çapaklardan kaynaklanıyor olabilir.",
        "action": "Taşıyıcı ruloları, kılavuz plakalarını ve hat üzerindeki temas noktalarını fiziksel çapak/aşınma açısından kontrol edin.",
    },
}

SYSTEM_PROMPT = """Sen otomotiv ve çelik yan sanayi kalite kontrol süreçlerinde uzman bir görsel muayene asistanısın.
Sana yalnızca bir YOLO nesne tespit modelinin ürettiği yapılandırılmış JSON çıktısı verilecek.

KESİN KURALLAR (Halüsinasyon Önleme):
1. SADECE JSON içindeki `summary_by_class` ve `detections` alanlarında yer alan kusur sınıflarını, adetlerini ve güven skorlarını kullan.
2. JSON'da yer almayan hiçbir kusur sınıfını, sayıyı veya görsel detayı ASLA uydurma (halüsinasyon yapma). Eğer `total_detections` 0 ise yalnızca kusur tespit edilmediğini belirt.
3. Kök neden analizinde kesin hüküm verme; her zaman ihtiyatlı bir mühendislik dili ("...kaynaklı olabilir", "...ihtimali değerlendirilmelidir") kullan.
4. Yanıtını Türkçe olarak ve tam olarak aşağıdaki 5 başlıkla üret:
   - **Özet**
   - **Tespit Edilen Kusurlar**
   - **Olası Kök Neden**
   - **Önerilen Aksiyon**
   - **Güven Notu**
"""


def build_user_prompt(payload: Dict[str, Any]) -> str:
    """Format the structured YOLO detection JSON into an explicit prompt for the LLM."""
    compact_payload = {
        "total_detections": payload.get("total_detections", 0),
        "confidence_threshold": payload.get("confidence_threshold", 0.25),
        "using_trained_neu_weights": payload.get("using_trained_neu_weights", True),
        "summary_by_class": payload.get("summary_by_class", {}),
        "detections": payload.get("detections", []),
    }
    json_str = json.dumps(compact_payload, ensure_ascii=False, indent=2)
    return (
        "Aşağıdaki YOLO tespit JSON verisini kullanarak kalite kontrol raporunu oluştur. "
        "JSON'da bulunmayan hiçbir kusuru rapora ekleme:\n\n"
        f"```json\n{json_str}\n```"
    )


def generate_rule_based_report(payload: Dict[str, Any], fallback_reason: str = "") -> str:
    """Generate a deterministic, structured Turkish quality report without calling an LLM."""
    total = int(payload.get("total_detections", 0))
    conf_thresh = float(payload.get("confidence_threshold", 0.25))
    summary_by_class: Dict[str, Dict[str, Any]] = payload.get("summary_by_class", {})
    trained_weights = bool(payload.get("using_trained_neu_weights", True))

    if total == 0 or not summary_by_class:
        weights_warning = (
            ""
            if trained_weights
            else " *(Uyarı: Henüz eğitilmiş `weights/best.pt` yüklenmediği için temel ağırlıklar kullanılmaktadır.)*"
        )
        note_suffix = f" ({fallback_reason})" if fallback_reason else ""
        return (
            "### Kalite Kontrol Raporu *(Kural Tabanlı Şablon)*\n\n"
            f"- **Özet:** Belirlenen güven eşiği (`{conf_thresh:.2f}`) üzerinde çelik sac yüzeyinde "
            "herhangi bir kusur tespit edilmemiştir.\n"
            "- **Tespit Edilen Kusurlar:** Tespit yok (`0` adet).\n"
            "- **Olası Kök Neden:** Yüzeyde raporlanacak bir anomali gözlenmemiştir.\n"
            "- **Önerilen Aksiyon:** Standart parti akışına devam edilebilir; şüpheli bir durum varsa "
            "güven eşiğini (confidence threshold) düşürerek tekrar inceleme yapılabilir.\n"
            f"- **Güven Notu:** Bu analiz yalnızca YOLO modelinin `{conf_thresh:.2f}` eşik değerindeki "
            f"çıktısına dayanmaktadır; aydınlatma ve kamera açısı farklılıkları sonuçları etkileyebilir.{weights_warning}{note_suffix}"
        )

    defect_lines = []
    cause_lines = []
    action_lines = []
    all_mean_confs = []

    for cls_name, stats in sorted(summary_by_class.items()):
        count = int(stats.get("count", 0))
        mean_c = float(stats.get("mean_confidence", 0.0))
        max_c = float(stats.get("max_confidence", 0.0))
        regions = ", ".join(stats.get("regions", [])) or "belirsiz"
        all_mean_confs.append(mean_c)

        info = DEFECT_KNOWLEDGE_TR.get(
            cls_name,
            {
                "label_tr": cls_name,
                "cause": "Üretim veya taşıma hattındaki yüzey etkileşimlerinden kaynaklanıyor olabilir.",
                "action": "İlgili üretim istasyonunda görsel doğrulama yapılması önerilir.",
            },
        )
        defect_lines.append(
            f"  - **{info['label_tr']} (`{cls_name}`):** {count} adet "
            f"(ortalama güven: `{mean_c:.2f}`, maks: `{max_c:.2f}`, bölge: `{regions}`)"
        )
        cause_lines.append(f"  - `{cls_name}`: {info['cause']}")
        action_lines.append(f"  - `{cls_name}`: {info['action']}")

    overall_conf = sum(all_mean_confs) / len(all_mean_confs) if all_mean_confs else 0.0
    conf_level = (
        "yüksek" if overall_conf >= 0.70 else ("orta" if overall_conf >= 0.45 else "düşük-orta")
    )
    distinct_classes = len(summary_by_class)

    return (
        "### Kalite Kontrol Raporu *(Kural Tabanlı Şablon)*\n\n"
        f"- **Özet:** İncelenen çelik sac yüzeyinde `{conf_thresh:.2f}` güven eşiği üzerinde "
        f"**{distinct_classes} farklı sınıfta toplam {total} adet** yüzey kusuru tespit edilmiştir.\n"
        "- **Tespit Edilen Kusurlar:**\n"
        + "\n".join(defect_lines)
        + "\n- **Olası Kök Neden:**\n"
        + "\n".join(cause_lines)
        + "\n- **Önerilen Aksiyon:**\n"
        + "\n".join(action_lines)
        + f"\n- **Güven Notu:** Tespitlerin ortalama güven skoru `{overall_conf:.2f}` ({conf_level} seviye) "
        "olup, bu değerlendirme bir ön tarama niteliğindedir; nihai kabul/ret kararı operatör "
        "muayenesiyle doğrulanmalıdır."
    )


def generate_quality_report(
    payload: Dict[str, Any],
    config_path: Path = Path("configs/config.yaml"),
) -> Tuple[str, str]:
    """Generate a quality report via Hugging Face Inference API or fall back to rule-based template.

    Returns:
        Tuple of (report_markdown, mode_used) where mode_used is either 'llm' or 'rule_based'.
    """
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        logger.info("HF_TOKEN environment variable not set. Using rule-based report generator.")
        return generate_rule_based_report(payload), "rule_based"

    cfg = load_config(config_path)
    llm_cfg = cfg.get("llm", {})
    model_id = str(llm_cfg.get("model_id", "Qwen/Qwen2.5-7B-Instruct"))
    max_new_tokens = int(llm_cfg.get("max_new_tokens", 450))
    temperature = float(llm_cfg.get("temperature", 0.2))
    timeout = float(llm_cfg.get("timeout_seconds", 20))

    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(model=model_id, token=hf_token, timeout=timeout)
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(payload)},
            ],
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        content = response.choices[0].message.content if response.choices else ""
        if content and content.strip():
            header = f"### Kalite Kontrol Raporu *(LLM: `{model_id}`)*\n\n"
            return header + content.strip(), "llm"

        logger.warning("Empty response from HF Inference API. Falling back to rule-based report.")
    except Exception as exc:
        logger.warning("HF Inference API error (%s). Falling back to rule-based report.", exc)

    return generate_rule_based_report(payload), "rule_based"
