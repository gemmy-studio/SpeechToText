from st_audiorec import st_audiorec
import streamlit as st
import whisper
import tempfile
from streamlit.components.v1 import html
import warnings
warnings.filterwarnings("ignore")

if 'last_uploaded' not in st.session_state:
    st.session_state.last_uploaded = None

if 'download_buttons_created' not in st.session_state:
    st.session_state.download_buttons_created = False


def main():
    st.set_page_config(page_title="Whisper 기반 Speech-to-Text 애플리케이션", page_icon="favicon.ico",
                       layout="wide", initial_sidebar_state="auto", menu_items=None)

    button = """
    <script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="woojae" data-color="#FFDD00" data-emoji="☕"  data-font="Cookie" data-text="Buy me a coffee" data-outline-color="#000000" data-font-color="#000000" data-coffee-color="#ffffff" ></script>
    """

    html(button, height=70, width=240)

    st.markdown(
        """
        <style>
            iframe[width="240"] {
                position: fixed;
                bottom: 30px;
                right: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.header("Whisper 기반 Speech-to-Text 애플리케이션")

    use_file_upload = "파일 업로드"
    use_file_record = "오디오 녹음"
    st.sidebar.write("## 사용할 옵션을 선택해주세요. 모델 사이즈의 크기는 최대 small까지 선택 가능합니다.")
    app_mode = st.sidebar.selectbox("오디오 업로드 방식", [
        use_file_upload, use_file_record])
    model_size = st.sidebar.selectbox("사용할 모델의 사이즈", [
        'tiny', 'base', 'small'], 1)

    if app_mode == use_file_upload:
        app_sst(
            str(model_size)
        )
    elif app_mode == use_file_record:
        app_sst_recoder(
            str(model_size)
        )


def app_sst(model_size: str):
    st.markdown(
        """
#### **파일 업로드를 통한 변환**

아래의 "파일 업로드" 버튼을 사용하여 원하는 오디오 파일을 업로드하세요. 지원하는 파일 형식은 mp3, wav, ogg, opus, m4a, flac 입니다. 파일을 업로드하면 자동으로 텍스트로 변환되며, 변환된 결과를 다운로드 받을 수 있습니다.
"""
    )

    uploaded_file = st.file_uploader(
        "파일 업로드", type=["mp3", "wav", "ogg", "opus", "m4a", "flac"])

    if uploaded_file:
        if st.session_state.last_uploaded != uploaded_file.name:
            st.session_state.last_uploaded = uploaded_file.name
            text_output = st.empty()

            # whisper 모델 로딩
            with st.spinner('STT 로딩중'):
                model = whisper.load_model(model_size)

            # 임시 파일에 업로드된 오디오 저장
            with st.spinner('STT 처리중'):
                with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmpfile:
                    tmpfile.write(uploaded_file.getvalue())
                    result = model.transcribe(tmpfile.name)
                    text = result["text"]
                    text_output.markdown(f"**Text:** {text}")

                    # 생성된 텍스트를 txt 파일로 다운로드 받을 수 있는 버튼 제공
                    st.download_button(
                        label="다운로드",
                        data=text.encode(),
                        file_name="output.txt",
                        mime="text/plain"
                    )
    else:
        st.session_state.last_uploaded = None


def app_sst_recoder(model_size: str):
    st.markdown(
        """
#### **직접 녹음하여 변환**

"Start Recording"과 "Stop" 버튼을 클릭하여 오디오 녹음을 시작하고 종료하세요. 녹음 완료 후 '변환하기' 버튼을 누르면 텍스트로 변환되며, 변환된 결과를 다운로드 받을 수 있습니다.
"""
    )

    recorded_file = False

    wav_audio_data = st_audiorec()
    if wav_audio_data is not None:
        recorded_file = True

    if recorded_file:  # recorded_file에 데이터가 있다면
        if st.button('변환하기'):
            text_output = st.empty()

            # whisper 모델 로딩
            with st.spinner('STT 로딩중'):
                model = whisper.load_model(model_size)

            # 임시 파일에 recorded_file 데이터 저장
            with st.spinner('STT 처리중'):
                with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmpfile:
                    # recorded_file 대신 wav_audio_data 사용
                    tmpfile.write(wav_audio_data)
                    result = model.transcribe(tmpfile.name)
                    text = result["text"]
                    text_output.markdown(f"**Text:** {text}")

                    # 생성된 텍스트를 txt 파일로 다운로드 받을 수 있는 버튼 제공
                    st.download_button(
                        label="다운로드",
                        data=text.encode(),
                        file_name="output.txt",
                        mime="text/plain"
                    )


if __name__ == "__main__":
    main()
