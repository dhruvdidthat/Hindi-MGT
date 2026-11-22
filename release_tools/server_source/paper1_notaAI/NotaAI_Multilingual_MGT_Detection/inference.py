import torch
import langid
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import LLMFeatureExtractor
from model import MGTDetectionModel

import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_text", type=str)
    parser.add_argument("--hf_token", type=str)

    args = parser.parse_args()
    return args


def get_models():
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    unseen_language_detection_model = MGTDetectionModel()
    unseen_language_detection_model.load_state_dict(
        torch.load(
            "MGTDetectionModel.pt",
            # weights_only=True,
            map_location=torch.device("cuda:0"),
        )
    )
    unseen_language_detection_model = unseen_language_detection_model.eval().to("cuda:0")
    # seen_language_detection_model = AutoModelForSequenceClassification.from_pretrained(
    #     "nota-ai/multilingual-e5-large-MGT-finetuned",
    #     device_map="auto",
    # ).eval()
    seen_language_detection_model = None

    model_dict = {"seen": seen_language_detection_model, "unseen": unseen_language_detection_model}

    return model_dict, tokenizer


def main(in_text, extractor: LLMFeatureExtractor):
    seen_lang = ["en", "ru", "de", "zh", "ar", "bg", "id", "ur", "it"]

    model_dict, tokenizer = get_models()

    input_text = in_text
    detected_language, _ = langid.classify(input_text)
    tokens = tokenizer(input_text, return_tensors="pt")

    if detected_language in seen_lang:
        tokens = tokens.to("cuda:0")
        with torch.no_grad():
            outputs = model_dict["seen"](**tokens)

    else:
        input_dict = extractor.feature_extract(input_text)

        input_tensors = {k: v.to("cuda:0") for k, v in input_dict.items()}
        with torch.no_grad():
            outputs = model_dict["unseen"](**input_tensors)

    print(f"outputs: {outputs}")
    logit = outputs
    pred = logit.argmax().item()
    print(f"Logits: {logit}")
    print(f"Probabilities: {torch.softmax(logit, dim=-1)}")
    print(f"Probabilities [1]: {torch.softmax(logit, dim=-1)[1]}")
    print(f"Probabilities [1].item(): {torch.softmax(logit, dim=-1)[1].item()}")
    print(f"pred before invert: {pred}")
    pred = 1 - pred  # 1= Human, 0= Machine
    res = "Human" if pred == 1 else "Machine"
    print(f"pred: {pred}")
    print(f"{input_text[:20]} > {res} generated text")


if __name__ == "__main__":
    # args = parse_args()
    in_text = "भारत में सेडैन सेगमेंट में मारुति सुजुकी की डिजायर कार लोगों को सबसे ज्यादा पसंद आती है। शानदार लुक और फीचर्स के साथ ही अच्छी माइलेज की वजह से इस सेडान की बंपर बिक्री होती है। आप भी मारुति सुजुकी डिजायर को महज एक लाख रुपये डाउनपेमेंट कर घर ले जा सकते हैं। फिलहाल आपको डिजायर के टॉप सेलिंग मॉडल डिजायर वीएक्सआई के साथ ही बाकी वेरिएंट्स की कीमतों के बारे में बताएं तो इस सेडान को LXi, VXi, ZXi और ZXi+ जैसे 4 ट्रिम लेवल के कुल 9 वेरिएंट्स में पेश किया गया है, जिनकी कीमतें 6.24 लाख रुपये से लेकर 9.18 लाख रुपये (एक्स शोरूम) तक है। पेट्रोल के साथ ही सीएनजी ऑप्शन में भी पेश यह 5 सीटर सेडान मैनुअल और ऑटोमैटिक ट्रांसमिशन ऑप्शन में है। डिजायर के पेट्रोल वेरिएंट्स की माइलेज 24.12 kmpl तक और सीएनजी वेरिएंट्स की माइलेज 31.12 km/kg तक की है।ये भी पढ़ें-आने वाली हैं मारुति सुजुकी और टाटा मोटर्स की नई सीएनजी कारें, हर दिन होगी बंपर बचतमारुति सुजुकी डिजायर वीएक्सआई मैनुअल लोन डाउनपेमेंट ईएमआई अब आपको मारुति सुजुकी डिजायर के टॉप सेलिंग वेरिएंट डिजायर वीएक्सआई मैनुअल फाइनैंस के बारे में बताएं तो इसकी एक्स शोरूम कीमत 6.24 लाख रुपये और ऑन-रोड प्राइस 8,18,043 रुपये है। आप अगर डिजायर वीएक्सआई को एक लाख रुपये डाउनपेमेंट (ऑन रोड + प्रोसेसिंग फी + फर्स्ट मंथ ईएमआई) कर फाइनैंस कराते हैं तो फिर आपको 7,18,043 रुपये लोन मिलेगा। आप अगर 5 साल तक के लिए 9 पर्सेंट ब्याज दर पर कार फाइनैंस कराते हैं तो फिर आपको अगले 60 महीनों तक के लिए हर महीने 14,905 रुपये ईएमआई के रूप में चुकाने होंगे। मारुति डिजायर वीएक्सआई मैनुअल को लोन लेकर खरीदने पर आपको 5 साल में करीब 1.76 लाख रुपये ब्याज लग जाएंगे।ये भी पढ़ें-इन 10 बॉलीवुड स्टार्स को कम दाम की कारों से है प्यार, लिस्ट में जॉन अब्राहम से लेकर सारा अली खान तकमारुति सुजुकी की टॉप सेलिंग कारों में से एकमारुति सुजुकी डिजायर वीएक्सआई ऑटोमैटिक लोन डाउनपेमेंट ईएमआई मारुति सुजुकी डिजायर वीएक्सआई ऑटोमैटिक की एक्स शोरूम कीमत 7.7 लाख रुपये और ऑन-रोड प्राइस 8,72,736 रुपये है। आप अगर डिजायर वीएक्सआई ऑटोमैटिक को एक लाख रुपये डाउनपेमेंट (ऑन रोड के साथ ही प्रोसेसिंग फी और फर्स्ट मंथ ईएमआई) कर फाइनैंस कराते हैं तो फिर आपको 7,72,736 रुपये लोन मिलेगा। आप अगर 5 साल तक के लिए 9 पर्सेंट ब्याज दर पर कार फाइनैंस कराते हैं तो फिर आपको अगले 60 महीनों तक के लिए हर महीने 16,041 रुपये ईएमआई के रूप में चुकाने होंगे। मारुति डिजायर वीएक्सआई ऑटोमैटिक को लोन लेकर खरीदने पर आपको 5 साल में करीब 1.9 लाख रुपये ब्याज लग जाएंगे।Disclaimer- डिजायर के ये दोनों वेरिएंट्स खरीदने से पहले आप नजदीकी मारुति सुजुकी डीलरशिप पर जाकर कार लोन और ईएमआई डिटेल्स के साथ ही ब्याज दर जरूर चेक कर लें।ये भी पढ़ें-खुशखबरी! अगले महीने भारतीय सड़कों पर दौड़ेंगी 3 नई इलेक्ट्रिक कारें, कम दाम में मिलेगा ज्यादा मजा"
    extractor = LLMFeatureExtractor("[redacted]")
    main(in_text, extractor)

