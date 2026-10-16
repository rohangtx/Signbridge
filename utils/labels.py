from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VOCAB=ROOT/'data'/'vocabulary_40.txt'
HINDI={'he':'वह','she':'वह','hello':'नमस्ते','goodbye':'अलविदा','thank_you':'धन्यवाद','sorry':'माफ़ कीजिए','please':'कृपया','yes':'हाँ','no':'नहीं','help':'मदद','stop':'रुको','okay':'ठीक है','me':'मैं','you':'आप','mother':'माँ','father':'पिता','brother':'भाई','sister':'बहन','friend':'दोस्त','teacher':'शिक्षक','student':'विद्यार्थी','home':'घर','school':'स्कूल','hospital':'अस्पताल','market':'बाज़ार','eat':'खाना','drink':'पीना','water':'पानी','food':'भोजन','tea':'चाय','come':'आना','go':'जाना','sit':'बैठना','stand':'खड़ा होना','read':'पढ़ना','write':'लिखना','what':'क्या','where':'कहाँ','when':'कब','today':'आज'}
def load_labels():
    if not VOCAB.exists():
        return []
    return [line.strip() for line in VOCAB.read_text(encoding='utf-8').splitlines() if line.strip()]
def display_label(x,lang='English'): return HINDI.get(x,x) if lang=='Hindi' else x.replace('_',' ').title()
