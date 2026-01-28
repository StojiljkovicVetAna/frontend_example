import streamlit as st
import cv2
import numpy as np
from PIL import Image
import dlib
from scipy.spatial import Delaunay
import io

st.set_page_config(page_title="Age Morph", layout="wide")

@st.cache_resource
def load_predictor():
    """Load dlib face detector and shape predictor"""
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    return detector, predictor

def get_landmarks(image, detector, predictor):
    """Extract facial landmarks from image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        return None
    
    shape = predictor(gray, faces[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    return landmarks

def add_boundary_points(landmarks, img_shape):
    """Add boundary points for full image morphing"""
    h, w = img_shape[:2]
    boundary = np.array([
        [0, 0], [w//2, 0], [w-1, 0],
        [0, h//2], [w-1, h//2],
        [0, h-1], [w//2, h-1], [w-1, h-1]
    ])
    return np.vstack([landmarks, boundary])

def morph_triangle(img1, img2, img, t1, t2, t, alpha):
    """Morph a single triangle"""
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))
    
    t1_rect = []
    t2_rect = []
    t_rect = []
    
    for i in range(3):
        t_rect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
    
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)
    
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2_rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    
    size = (r[2], r[3])
    warp_mat1 = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t_rect))
    warp_mat2 = cv2.getAffineTransform(np.float32(t2_rect), np.float32(t_rect))
    
    img1_rect_warped = cv2.warpAffine(img1_rect, warp_mat1, size, None, 
                                       flags=cv2.INTER_LINEAR, 
                                       borderMode=cv2.BORDER_REFLECT_101)
    img2_rect_warped = cv2.warpAffine(img2_rect, warp_mat2, size, None,
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT_101)
    
    img_rect = (1.0 - alpha) * img1_rect_warped + alpha * img2_rect_warped
    
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * (1 - mask) + img_rect * mask

def morph_faces(img1, img2, landmarks1, landmarks2, alpha):
    """Morph between two faces"""
    points1 = add_boundary_points(landmarks1, img1.shape)
    points2 = add_boundary_points(landmarks2, img2.shape)
    points = (1 - alpha) * points1 + alpha * points2
    
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    img_morphed = np.zeros(img1.shape, dtype=img1.dtype)
    
    tri = Delaunay(points)
    
    for simplex in tri.simplices:
        t1 = points1[simplex]
        t2 = points2[simplex]
        t = points[simplex]
        morph_triangle(img1, img2, img_morphed, t1, t2, t, alpha)
    
    return np.uint8(img_morphed)

def main():
    st.title("🎭 Age Morphing App")
    st.write("Upload photos of an older and younger person, then morph between them")
    
    # Try to load predictor with error handling
    try:
        detector, predictor = load_predictor()
    except Exception as e:
        st.error(f"Failed to load face predictor: {str(e)}")
        st.info("Make sure 'shape_predictor_68_face_landmarks.dat' exists in the current directory")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Older Photo")
        older_file = st.file_uploader("Upload older person's photo", type=['jpg', 'jpeg', 'png'], key='older')
    
    with col2:
        st.subheader("Younger Photo")
        younger_file = st.file_uploader("Upload younger person's photo", type=['jpg', 'jpeg', 'png'], key='younger')
    
    if older_file and younger_file:
        # Load images
        older_img = Image.open(older_file)
        younger_img = Image.open(younger_file)
        
        older_cv = cv2.cvtColor(np.array(older_img), cv2.COLOR_RGB2BGR)
        younger_cv = cv2.cvtColor(np.array(younger_img), cv2.COLOR_RGB2BGR)
        
        # Resize to same dimensions
        h, w = min(older_cv.shape[0], younger_cv.shape[0]), min(older_cv.shape[1], younger_cv.shape[1])
        older_cv = cv2.resize(older_cv, (w, h))
        younger_cv = cv2.resize(younger_cv, (w, h))
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(older_img, caption="Older", use_container_width=True)
        with col2:
            st.image(younger_img, caption="Younger", use_container_width=True)
        
        try:
            with st.spinner("Detecting faces..."):
                landmarks_older = get_landmarks(older_cv, detector, predictor)
                landmarks_younger = get_landmarks(younger_cv, detector, predictor)
            
            if landmarks_older is None or landmarks_younger is None:
                st.error("Could not detect face in one or both images. Please use clear frontal face photos.")
                return
            
            st.success("Faces detected!")
            
            st.subheader("Morph Control")
            alpha = st.slider("Morph amount (0 = older, 1 = younger)", 0.0, 1.0, 0.0, 0.05)
            
            if st.button("Generate Morph", type="primary"):
                with st.spinner("Morphing..."):
                    morphed = morph_faces(older_cv, younger_cv, landmarks_older, landmarks_younger, alpha)
                    morphed_rgb = cv2.cvtColor(morphed, cv2.COLOR_BGR2RGB)
                    
                    st.subheader("Morphed Result")
                    st.image(morphed_rgb, use_container_width=True)
                    
                    # Provide download
                    result_pil = Image.fromarray(morphed_rgb)
                    buf = io.BytesIO()
                    result_pil.save(buf, format='PNG')
                    st.download_button(
                        label="Download Morphed Image",
                        data=buf.getvalue(),
                        file_name=f"morphed_{alpha:.2f}.png",
                        mime="image/png"
                    )
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Make sure you have downloaded the shape_predictor_68_face_landmarks.dat file")

if __name__ == "__main__":
    main()