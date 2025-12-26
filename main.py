import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import tempfile
import os
from io import BytesIO
import gc

# Import trực tiếp để tránh lỗi AttributeError: module 'mediapipe' has no attribute 'solutions'
try:
    import mediapipe as mp
    from mediapipe.python.solutions import face_detection as mp_face
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    USE_AI = True
except (ImportError, AttributeError):
    USE_AI = False

# --- CÁC HÀM XỬ LÝ LOGIC ---

def calculate_sharpness(frame):
    """Tính độ sắc nét bằng biến thiên Laplacian"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def get_ai_scores(frame, face_model, pose_model):
    """Chấm điểm khuôn mặt và tư thế sử dụng MediaPipe"""
    face_score = 0.0
    num_faces = 0
    pose_score = 0.0
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 1. Chấm điểm khuôn mặt
    face_results = face_model.process(rgb_frame)
    if face_results.detections:
        num_faces = len(face_results.detections)
        face_score = np.mean([d.score[0] for d in face_results.detections])
    
    # 2. Chấm điểm tư thế (Pose) - Quan trọng cho dancer
    pose_results = pose_model.process(rgb_frame)
    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark
        # Tính độ mở của tay (điểm 15, 16) và chân (điểm 27, 28)
        arm_span = np.sqrt((lm[15].x - lm[16].x)**2 + (lm[15].y - lm[16].y)**2)
        leg_span = np.sqrt((lm[27].x - lm[28].x)**2 + (lm[27].y - lm[28].y)**2)
        # Điểm pose cao khi cơ thể bung tỏa (extension)
        pose_score = min((arm_span + leg_span) * 1.5, 1.0)
        
    return face_score, num_faces, pose_score

def calculate_total_score(frame, face_model, pose_model):
    """Tính điểm tổng hợp, kiểm tra an toàn nếu AI model không tồn tại"""
    sharpness = calculate_sharpness(frame)
    norm_sharpness = min(sharpness / 800.0, 1.0)
    
    # Kiểm tra nếu cả hai model đều sẵn sàng
    if USE_AI and face_model is not None and pose_model is not None:
        face_score, num_faces, pose_score = get_ai_scores(frame, face_model, pose_model)
        
        if num_faces > 0:
            total = (face_score * 0.5) + (norm_sharpness * 0.3) + (pose_score * 0.2)
        else:
            total = (pose_score * 0.6) + (norm_sharpness * 0.4)
            
        return total, sharpness, face_score, pose_score, num_faces
    else:
        # Chế độ dự phòng nếu AI lỗi: Chỉ tính dựa trên độ nét
        return norm_sharpness, sharpness, 0.0, 0.0, 0

def extract_best_frames(video_path, num_frames=15, sample_rate=5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_scores = []
    
    # Khởi tạo giá trị mặc định để tránh lỗi UnboundLocalError
    face_model = None
    pose_model = None
    
    # Chỉ khởi tạo model nếu thư viện MediaPipe khả dụng
    if USE_AI:
        try:
            face_model = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
            pose_model = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        except Exception as e:
            st.warning(f"Không thể khởi động AI Model: {e}. Hệ thống sẽ dùng chế độ quét cơ bản.")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if count % sample_rate == 0:
                # Truyền các model (có thể là None) vào hàm tính điểm
                score, sharp, face, pose, n_faces = calculate_total_score(frame, face_model, pose_model)
                
                frame_scores.append({
                    'frame': frame.copy(),
                    'score': score,
                    'timestamp': count / fps,
                    'details': f"Nét: {sharp:.0f} | Mặt: {face:.2f} | Dáng: {pose:.2f}"
                })
                
                progress = min(count / total_frames, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"⚡ Đang phân tích: {count}/{total_frames} frames")

            count += 1
            if count % 150 == 0: gc.collect()
            
    finally:
        cap.release()
        # Chỉ đóng nếu model đã được khởi tạo
        if face_model: face_model.close()
        if pose_model: pose_model.close()
        progress_bar.empty()
        status_text.empty()

    # Sắp xếp và lọc các frame quá gần nhau (trong vòng 0.8s)
    frame_scores.sort(key=lambda x: x['score'], reverse=True)
    
    final_selection = []
    for f in frame_scores:
        if not any(abs(f['timestamp'] - s['timestamp']) < 0.8 for s in final_selection):
            final_selection.append(f)
        if len(final_selection) >= num_frames: break
            
    return final_selection

# --- GIAO DIỆN STREAMLIT ---

def main():
    st.set_page_config(page_title="Dance Best Frame AI", layout="wide")
    
    st.title("🕺 AI Dance Best Frame")
    st.markdown("Hệ thống tự động chấm điểm **Nét + Mặt + Tư thế nhảy**")

    with st.sidebar:
        st.header("⚙️ Tùy chỉnh AI")
        num_frames = st.slider("Số lượng ảnh muốn lấy", 5, 30, 12)
        sample_rate = st.select_slider("Độ chi tiết quét (càng thấp càng chậm)", options=[2, 5, 10, 20], value=5)
        upscale = st.checkbox("Tự động Upscale 2x (LANCZOS)", value=True)
        st.info("Lưu ý: Quét chi tiết (2-5) sẽ tốn RAM hơn.")

    uploaded_file = st.file_uploader("Tải video của bạn lên", type=['mp4', 'mov', 'avi'])

    if uploaded_file:
        # Tạo file tạm nhưng không tự động xóa (để OpenCV đọc được)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            temp_path = tfile.name # Lưu lại đường dẫn

        try:
            # Thực hiện xử lý
            results = extract_best_frames(temp_path, num_frames, sample_rate)
            
            if results:
                st.success(f"Đã lọc ra {len(results)} khoảnh khắc đẹp nhất!")
                
                # Hiển thị Grid (giữ nguyên logic hiển thị của bạn)
                cols = st.columns(3)
                for i, data in enumerate(results):
                    with cols[i % 3]:
                        img_rgb = cv2.cvtColor(data['frame'], cv2.COLOR_BGR2RGB)
                        st.image(img_rgb, caption=f"Top {i+1}")
                        
                        pil_img = Image.fromarray(img_rgb)
                        if upscale:
                            new_size = (pil_img.width * 2, pil_img.height * 2)
                            pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
                        
                        buf = BytesIO()
                        pil_img.save(buf, format="JPEG", quality=95)
                        st.download_button(
                            label=f"Tải ảnh {i+1}",
                            data=buf.getvalue(),
                            file_name=f"best_frame_{i+1}.jpg",
                            key=f"btn_{i}" # Thêm key để tránh trùng lặp
                        )
        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")
        finally:
            # QUAN TRỌNG: Giải phóng bộ nhớ và xóa file tạm an toàn
            gc.collect()
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except PermissionError:
                    # Nếu vẫn kẹt, Windows sẽ tự xóa khi app đóng hoặc lần chạy sau
                    pass

if __name__ == "__main__":
    main()