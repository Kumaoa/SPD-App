import cv2
import dlib
import os
import numpy as np
from tkinter import *
from tkinter import filedialog, messagebox  
from PIL import Image, ImageTk
import webbrowser

# Function to load images from a directory
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images

# Function to recognize faces in an image
def recognize_faces(image, detector, face_recognizer):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    face_encodings = []
    for face in faces:
        # Compute face descriptor for each face
        shape = predictor(gray, face)
        face_descriptor = face_recognizer.compute_face_descriptor(image, shape)
        face_encodings.append(face_descriptor)
    return face_encodings

# Function to compare face encodings
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.5):
    distances = np.linalg.norm(np.array(known_face_encodings) - np.array(face_encoding_to_check), axis=1)
    return [distance <= tolerance for distance in distances]

# Function to display images in a grid layout
def display_images(images):
    num_images = len(images)
    num_cols = int(num_images ** 0.5)
    num_rows = (num_images + num_cols - 1) // num_cols

    root = Toplevel()
    root.title("Images with Person")
    root.geometry("800x600")  # Set a fixed size for the window

    # Calculate the target size for the images
    target_size = (300, 300)  # Adjust the dimensions as needed

    # Create a frame to contain the images
    frame = Frame(root)
    frame.pack(fill=BOTH, expand=YES)

    # Create a canvas with scrollbars
    canvas = Canvas(frame)
    canvas.pack(side=LEFT, fill=BOTH, expand=YES)

    scrollbar = Scrollbar(frame, orient=VERTICAL, command=canvas.yview)
    scrollbar.pack(side=RIGHT, fill=Y)

    canvas.configure(yscrollcommand=scrollbar.set)

    # Create another frame inside the canvas to hold the images
    inner_frame = Frame(canvas)
    canvas.create_window((0, 0), window=inner_frame, anchor=NW)

    # Add images to the inner frame
    for i, img in enumerate(images):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(img, target_size)  # Resize the image
        photo = ImageTk.PhotoImage(image=Image.fromarray(resized_img))
        label = Label(inner_frame, image=photo)
        label.image = photo  # Keep a reference to prevent garbage collection
        label.grid(row=i // num_cols, column=i % num_cols, padx=5, pady=5)

    # Configure scrollable region
    inner_frame.update_idletasks()  # Update inner frame to compute its size
    canvas.config(scrollregion=canvas.bbox("all"))

    # Add a button to save images
    def save_images():
        folder_selected = filedialog.askdirectory(title="Select Destination Folder")
        if folder_selected:
            for i, img in enumerate(images):
                filename = os.path.join(folder_selected, f"image_{i}.jpg")
                cv2.imwrite(filename, img)
            messagebox.showinfo("Save Successful", "Images saved successfully.")

    save_button = Button(root, text="Save Images", command=save_images)
    save_button.pack(pady=10)

    root.mainloop()
    
# Function to handle button click to select selfie photo
def select_selfie_photo():
    global selfie_image
    # Show loading message
    loading_label = Label(root, text="Please wait, processing...", font=("Arial", 12))
    loading_label.pack(pady=10)
    root.update()  # Update the window to display the loading message
    
    selfie_path = filedialog.askopenfilename(title="Select Selfie Photo", filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))
    if selfie_path:
        selfie_image = cv2.imread(selfie_path)
        images_with_person = find_images_with_person(selfie_image)
        if images_with_person:
            # Remove loading message
            loading_label.pack_forget()
            display_images(images_with_person)
        else:
            loading_label.pack_forget()  # Remove loading message
            messagebox.showinfo("No Images Found", "No images with the person from the selfie found.")


# Function to find images with the person from the selfie
def find_images_with_person(selfie_image):
    selfie_face_encodings = recognize_faces(selfie_image, detector, face_recognizer)
    images_with_person = []
    for img in all_images:
        face_encodings = recognize_faces(img, detector, face_recognizer)
        for face_encoding in face_encodings:
            if any(compare_faces(selfie_face_encodings, face_encoding)):
                images_with_person.append(img)
                break  # Once the person is detected, no need to check further in the same image
    return images_with_person

# Function to open Google Form
def open_google_form():
    url = "https://docs.google.com/forms/d/1Qe8-dZtv1DzFmokpinVEh6BKQrvrAy_FSkHVXeBVBoI/edit#responses"  # Replace with your Google Form URL
    webbrowser.open_new(url)

# Function to display contact information
def display_contact_info():
    messagebox.showinfo("Contact Us", "SPD App By:\nSami 442811408\nKhalid 442803129\nTareq 442811972")

if __name__ == "__main__":
    # Folder containing images
    folder_path = r"Database"

    # Load all images from the folder
    all_images = load_images_from_folder(folder_path)

    # Initialize dlib's face detector and face recognition model
    detector = dlib.get_frontal_face_detector()
    face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    # Initialize dlib's shape predictor
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)

    # Create a Tkinter window
    root = Tk()
    root.title("SPD App")
    root.geometry("800x600")

    # Load background photo
    background_photo_path = "background.png"  # Path to your background photo
    background_photo = PhotoImage(file=background_photo_path)
    background_label = Label(root, image=background_photo)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    # Create a label
    label = Label(root, text="Welcome To SPD App", font=("Fugaz One", 28))
    label.pack(pady=10)

    # Create a button to select selfie photo
    select_button = Button(root, text="Select Photo", font=("Fugaz One", 14), command=select_selfie_photo)
    select_button.pack(pady=5)

    # Create a button to open Google Form
    form_button = Button(root, text="Open Clients Requests", font=("Fugaz One", 14), command=open_google_form)
    form_button.pack(pady=5)

    # Create a button to display contact information
    contact_button = Button(root, text="Contact Us", font=("Fugaz One", 14), command=display_contact_info)
    contact_button.pack(pady=5)

    # Run the Tkinter event loop
    root.mainloop()
