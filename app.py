import time
import gradio as gr
import utils
import commons
from models import SynthesizerTrn
from text import text_to_sequence
from torch import no_grad, LongTensor
import torch

hps_ms = utils.get_hparams_from_file(r'./model/config.json')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net_g_ms = SynthesizerTrn(
    len(hps_ms.symbols),
    hps_ms.data.filter_length // 2 + 1,
    hps_ms.train.segment_size // hps_ms.data.hop_length,
    n_speakers=hps_ms.data.n_speakers,
    **hps_ms.model).to(device)
_ = net_g_ms.eval()
speakers = hps_ms.speakers
model, optimizer, learning_rate, epochs = utils.load_checkpoint(r'./model/G_953000.pth', net_g_ms, None)

def get_text(text, hps):
    text_norm, clean_text = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm, clean_text

def vits(text, language, speaker_id, noise_scale, noise_scale_w, length_scale):
    start = time.perf_counter()
    if not len(text):
        return "入力文が空", None, None
    text = text.replace('\n', ' ').replace('\r', '').replace(" ", "")
    if len(text) > 550:
        return f"入力文が超過！{len(text)}>120", None, None
    if language == 0:
        text = f"[ZH]{text}[ZH]"
    elif language == 1:
        text = f"[JA]{text}[JA]"
    else:
        text = f"{text}"
    stn_tst, clean_text = get_text(text, hps_ms)
    with no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = LongTensor([stn_tst.size(0)])
        speaker_id = LongTensor([speaker_id])
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=speaker_id, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                               length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()

    return "生成成功!", (22050, audio), f"生成耗时 {round(time.perf_counter()-start, 2)} s"

def search_speaker(search_value):
    for s in speakers:
        if search_value == s:
            return s
    for s in speakers:
        if search_value in s:
            return s

def change_lang(language):
    if language == 0:
        return 0.6, 0.668, 1.2
    else:
        return 0.6, 0.668, 1.1

download_audio_js = """
() =>{{
    let root = document.querySelector("body > gradio-app");
    if (root.shadowRoot != null)
        root = root.shadowRoot;
    let audio = root.querySelector("#tts-audio").querySelector("audio");
    let text = root.querySelector("#input-text").querySelector("textarea");
    if (audio == undefined)
        return;
    text = text.value;
    if (text == undefined)
        text = Math.floor(Math.random()*100000000);
    audio = audio.src;
    let oA = document.createElement("a");
    oA.download = text.substr(0, 20)+'.wav';
    oA.href = audio;
    document.body.appendChild(oA);
    oA.click();
    oA.remove();
}}
"""

if __name__ == '__main__':
    with gr.Blocks() as app:
        gr.Markdown(
            "# <center> <a href='https://www.instagram.com/wifi6ghz/reels/'>AIテイオー demo\n"
            "<div align='center'><p><img width='200' height='200' alt='trsrtr_instagram' \n"
            "src='https://static.xx.fbcdn.net/rsrc.php/v3/yx/r/tBxa1IFcTQH.png'><img alt='btc' width='200' height='200'  \n"
            "src='https://pxchk20240406html.pages.dev/donateQR_btflr.png'></p>主要ウマ娘日本語，原神中国語，原神日本語，崩壊スターレイル3</a></div>"
        )

        with gr.Tabs():
            with gr.TabItem("vits"):
                with gr.Row():
                    with gr.Column():
                        input_text = gr.Textbox(label="Text (110 words limitation)", lines=5, value="やりますねぇ！。イキ過ぎぃ！！イクイクッ！。胸にかけて、胸に！", elem_id=f"input-text")
                        lang = gr.Dropdown(label="Language", choices=["中国語", "日本語", "中日混合"],
                                    type="index", value="日本語")
                        btn = gr.Button(value="Submit")
                        with gr.Row():
                            search = gr.Textbox(label="Search Speaker", lines=1)
                            btn2 = gr.Button(value="Search")
                        sid = gr.Dropdown(label="Speaker", choices=speakers, type="index", value=speakers[20])
                        with gr.Row():
                            ns = gr.Slider(label="noise(感情の変化の程度)", minimum=0.1, maximum=1.0, step=0.1, value=0.7, interactive=True)
                            nsw = gr.Slider(label="noise_w(音素の発音の長さ)", minimum=0.1, maximum=1.0, step=0.1, value=0.7, interactive=True)
                            ls = gr.Slider(label="length(全体の長さ)", minimum=0.1, maximum=2.0, step=0.1, value=1.3, interactive=True)
                    with gr.Column():
                        o1 = gr.Textbox(label="Output Message")
                        o2 = gr.Audio(label="Output Audio", elem_id=f"tts-audio")
                        o3 = gr.Textbox(label="Extra Info")
                        download = gr.Button("Download Audio")
                    btn.click(vits, inputs=[input_text, lang, sid, ns, nsw, ls], outputs=[o1, o2, o3], api_name="generate")
                    download.click(None, [], [], _js=download_audio_js.format())
                    btn2.click(search_speaker, inputs=[search], outputs=[sid])
                    lang.change(change_lang, inputs=[lang], outputs=[ns, nsw, ls])
            with gr.TabItem("使用可能人物一覧(翻訳遅れのため中国語)"):
                gr.Radio(label="Speaker", choices=speakers, interactive=False, type="index")
    app.queue(concurrency_count=1).launch()