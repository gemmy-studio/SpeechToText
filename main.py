from audio_recorder_streamlit import audio_recorder
import streamlit as st
import logging
import logging.handlers
import whisper
import tempfile
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

if 'last_uploaded' not in st.session_state:
    st.session_state.last_uploaded = None

if 'download_buttons_created' not in st.session_state:
    st.session_state.download_buttons_created = False


def main():
    st.header("Speech-to-Text 오디오를 텍스트로 변환하기")

    use_file_upload = "파일 업로드"
    use_file_record = "오디오 녹음"
    app_mode = st.selectbox("오디오 업로드 방식을 선택해주세요.", [
                            use_file_upload, use_file_record])
    model_size = st.selectbox("사용할 모델의 사이즈를 선택해주세요.", [
        'tiny', 'base', 'small', 'medium'])

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

마이크 모양 버튼을 클릭하여 직접 오디오를 녹음을 시작하고 종료하세요. 녹음이 완료되면 자동으로 텍스트로 변환되며, 변환된 결과를 다운로드 받을 수 있습니다.
"""
    )

    recorded_file = False
    audio_bytes = audio_recorder()
    if audio_bytes:
        recorded_file = st.audio(audio_bytes, format="audio/wav")

    if recorded_file:  # recorded_file에 데이터가 있다면
        text_output = st.empty()

        # whisper 모델 로딩
        with st.spinner('STT 로딩중'):
            model = whisper.load_model(model_size)

        # 임시 파일에 recorded_file 데이터 저장
        with st.spinner('STT 처리중'):
            with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmpfile:
                tmpfile.write(audio_bytes)  # recorded_file 대신 audio_bytes 사용
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


if __name__ == "__main__":

    main()