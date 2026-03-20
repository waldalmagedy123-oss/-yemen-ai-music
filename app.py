import streamlit as st
from audiocraft.models import MusicGen
import torch
import scipy.io.wavfile
import io
import numpy as np

# إعدادات الصفحة (تظهر في تبويب المتصفح)
st.set_page_config(page_title="Yemen AI Music", page_icon="🎵")

# تصميم الواجهة
st.title("مولد الموسيقى الذكي - اليمن 🇾🇪")
st.markdown("""
مرحباً بك في النسخة التجريبية من موقع توليد الموسيقى بالذكاء الاصطناعي. 
اكتب وصفاً للموسيقى التي تتخيلها وسيقوم النظام بإنشائها لك.
""")

# خانة إدخال النص
prompt = st.text_input("صف الموسيقى (بالإنجليزية يفضل حالياً):", placeholder="e.g. Traditional Yemeni Oud with modern techno beat")

# منزلق لتحديد مدة المقطع
duration = st.slider("مدة المقطع (بالثواني):", min_value=2, max_value=10, value=5)

# زر التوليد
if st.button("إيقاع الموسيقى الآن ✨"):
    if prompt:
        try:
            with st.spinner("جاري معالجة الإشارة الصوتية... قد يستغرق ذلك دقيقة:"):
                # تحميل النموذج الصغير لضمان السرعة على السيرفرات المجانية
                model = MusicGen.get_pretrained('facebook/musicgen-small')
                model.set_generation_params(duration=duration)
                
                # توليد الصوت
                wav = model.generate([prompt])
                
                # معالجة الملف الصوتي للعرض
                sampling_rate = 32000
                audio_cpu = wav[0].cpu().numpy()[0]
                
                # تحويل إلى Bytes ليتمكن المتصفح من قراءته
                byte_io = io.BytesIO()
                scipy.io.wavfile.write(byte_io, sampling_rate, audio_cpu)
                
                # عرض النتيجة
                st.audio(byte_io, format="audio/wav")
                st.success("تم التوليد بنجاح! يمكنك سماع المقطع أعلاه.")
                
                # زر للتحميل
                st.download_button(label="تحميل المقطع ⬇️", 
                                   data=byte_io.getvalue(), 
                                   file_name="yemen_ai_music.wav", 
                                   mime="audio/wav")
        except Exception as e:
            st.error(f"حدث خطأ تقني: {e}")
    else:
        st.warning("الرجاء كتابة وصف للموسيقى أولاً.")

st.info("ملاحظة للمهندس علي: هذا الموقع يعمل حالياً على معالج السيرفر المستضيف (CPU/GPU).")

