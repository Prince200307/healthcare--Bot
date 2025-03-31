from tkinter import *
from tkinter import ttk, messagebox, scrolledtext
import os
import webbrowser
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree
from difflib import get_close_matches

# Modern color scheme
PRIMARY = "#2B5876"
SECONDARY = "#4E4376"
ACCENT = "#00C9FF"
WHITE = "#FFFFFF"
LIGHT_GRAY = "#F5F7FA"
DARK_GRAY = "#2D3748"
SUCCESS = "#48BB78"
WARNING = "#ED8936"
ERROR = "#F56565"

class HyperlinkManager:
    def __init__(self, text):
        self.text = text
        self.text.tag_config("hyper", foreground=ACCENT, underline=1)
        self.text.tag_bind("hyper", "<Enter>", self._enter)
        self.text.tag_bind("hyper", "<Leave>", self._leave)
        self.text.tag_bind("hyper", "<Button-1>", self._click)
        self.reset()

    def reset(self):
        self.links = {}

    def add(self, action):
        tag = "hyper-%d" % len(self.links)
        self.links[tag] = action
        return "hyper", tag

    def _enter(self, event):
        self.text.config(cursor="hand2")

    def _leave(self, event):
        self.text.config(cursor="")

    def _click(self, event):
        for tag in self.text.tag_names(CURRENT):
            if tag[:6] == "hyper-":
                self.links[tag]()
                return

class HealthcareChatbot:
    def __init__(self):
        # Load datasets
        self.training_dataset = pd.read_csv('Training.csv')
        self.test_dataset = pd.read_csv('Testing.csv')
        
        # Prepare data
        self.X = self.training_dataset.iloc[:, 0:132].values
        self.Y = self.training_dataset.iloc[:, -1].values
        
        # Dimensionality reduction
        self.dimensionality_reduction = self.training_dataset.groupby(self.training_dataset['prognosis']).max()
        
        # Encode labels
        self.labelencoder = LabelEncoder()
        self.y = self.labelencoder.fit_transform(self.Y)
        
        # Train classifier
        self.classifier = DecisionTreeClassifier()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.25, random_state=0)
        self.classifier.fit(self.X_train, self.y_train)
        
        # Prepare symptom list
        self.cols = self.training_dataset.columns[:-1]
        self.all_symptoms = list(self.cols)
        
        # Load doctors data
        self.doctors = pd.read_csv('doctors_dataset.csv', names=['Name', 'Description'])
        self.diseases = pd.DataFrame(self.dimensionality_reduction.index)
        self.doctors['disease'] = self.diseases['prognosis']
        
        # Initialize GUI
        self.root = Tk()
        self.root.title("AI Healthcare Chatbot")
        self.root.geometry("1200x800")
        self.root.configure(bg=WHITE)
        
        # Make window resizable
        self.root.minsize(1000, 700)
        
        # Custom fonts - made bolder
        self.title_font = ('Helvetica', 24, 'bold')
        self.subtitle_font = ('Helvetica', 18, 'bold')
        self.body_font = ('Helvetica', 14, 'bold')
        self.button_font = ('Helvetica', 14, 'bold')
        
        # Create container frame
        self.container = Frame(self.root)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        
        # Initialize pages
        self.frames = {}
        for F in (MainPage, LoginPage, RegisterPage, ChatbotPage, TraditionalDiagnosisPage):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        
        # Show main page first
        self.show_frame("MainPage")
        
        # Status bar
        self.status_var = StringVar()
        self.status_var.set("Ready")
        self.status_bar = Label(self.root, textvariable=self.status_var, 
                              relief=SUNKEN, anchor=W, padx=10, pady=5,
                              font=self.body_font, bg=LIGHT_GRAY)
        self.status_bar.pack(side=BOTTOM, fill=X)
    
    def show_frame(self, page_name):
        """Show a frame for the given page name"""
        frame = self.frames[page_name]
        frame.tkraise()
    
    def update_status(self, message):
        """Update status bar message"""
        self.status_var.set(message)
    
    def authenticate(self, username, password):
        """Authenticate user login"""
        if not username or not password:
            self.update_status("Please enter both username and password")
            return False
            
        list_of_files = os.listdir()
        if username in list_of_files:
            with open(username, "r") as file:
                verify = file.read().splitlines()
                if password in verify:
                    self.update_status("Login successful")
                    return True
                else:
                    self.update_status("Incorrect password")
                    return False
        else:
            self.update_status("User not found. Please register.")
            return False
    
    def register(self, username, password):
        """Register new user"""
        if not username or not password:
            self.update_status("Please enter both username and password")
            return False
            
        if username in os.listdir():
            self.update_status("Username already exists")
            return False
            
        with open(username, "w") as file:
            file.write(f"{username}\n{password}")
        
        self.update_status("Registration successful")
        return True
    
    def analyze_symptoms(self, symptoms):
        """Analyze the entered symptoms and provide diagnosis"""
        if not symptoms:
            self.update_status("Please add at least one symptom")
            return None
            
        try:
            # Convert symptoms to feature vector
            symptom_indices = [self.all_symptoms.index(symptom) for symptom in symptoms 
                            if symptom in self.all_symptoms]
            
            if not symptom_indices:
                self.update_status("No valid symptoms found for analysis")
                return None
                
            # Create feature vector
            X = np.zeros(len(self.all_symptoms))
            for idx in symptom_indices:
                X[idx] = 1
                
            # Predict disease
            prediction = self.classifier.predict([X])
            disease = self.labelencoder.inverse_transform(prediction)[0]
            
            # Get disease info
            disease_data = self.dimensionality_reduction.loc[disease]
            symptoms_given = [col for col, val in zip(self.cols, disease_data) if val == 1]
            
            # Calculate confidence level
            confidence_level = len(set(symptoms) & set(symptoms_given)) / len(symptoms_given) if len(symptoms_given) > 0 else 0.0
            
            # Doctor recommendation
            row = self.doctors[self.doctors['disease'] == disease]
            
            result = {
                "disease": disease,
                "symptoms_present": symptoms,
                "symptoms_given": symptoms_given,
                "confidence": confidence_level,
                "doctor": row['Name'].values[0] if not row.empty else None,
                "doctor_link": row['Description'].values[0] if not row.empty else None
            }
            
            return result
            
        except Exception as e:
            self.update_status(f"Error during diagnosis: {str(e)}")
            return None
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

class MainPage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg=LIGHT_GRAY)
        self.controller = controller
        
        # Content frame
        content_frame = Frame(self, bg=LIGHT_GRAY)
        content_frame.pack(expand=True, fill=BOTH, padx=20, pady=20)
        
        # Header
        header_frame = Frame(content_frame, bg=PRIMARY)
        header_frame.pack(fill=X, pady=(0, 30))
        
        Label(header_frame, text="AI Healthcare Diagnostic Assistant", 
             font=controller.title_font, bg=PRIMARY, fg=WHITE, padx=20, pady=10).pack(fill=X)
        
        # Main content
        main_content = Frame(content_frame, bg=LIGHT_GRAY)
        main_content.pack(expand=True, fill=BOTH)
        
        # Left side - description
        left_frame = Frame(main_content, bg=LIGHT_GRAY, padx=20, pady=20)
        left_frame.pack(side=LEFT, fill=BOTH, expand=True)
        
        Label(left_frame, text="Your Personal", 
             font=controller.subtitle_font, bg=LIGHT_GRAY).pack(pady=5)
        Label(left_frame, text="Healthcare Assistant", 
             font=controller.title_font, fg=ACCENT, bg=LIGHT_GRAY).pack(pady=5)
        
        Label(left_frame, 
             text="Our AI-powered chatbot provides preliminary medical diagnosis\n"
                  "based on your symptoms and connects you with specialists.\n\n"
                  "Get instant insights about potential health conditions and\n"
                  "find the right medical professionals for your needs.",
             font=controller.body_font,
             bg=LIGHT_GRAY,
             wraplength=400, justify=LEFT).pack(pady=20)
        
        # Feature highlights
        features = [
            "✓ Symptom-based diagnosis",
            "✓ Doctor recommendations",
            "✓ Confidence level scoring",
            "✓ Traditional diagnosis mode"
        ]
        
        for feature in features:
            Label(left_frame, text=feature, font=controller.body_font, bg=LIGHT_GRAY, anchor="w").pack(fill=X, pady=5)
        
        # Right side - image placeholder
        right_frame = Frame(main_content, bg=LIGHT_GRAY)
        right_frame.pack(side=RIGHT, fill=BOTH, padx=20)
        
        try:
            img = Image.open("healthcare_icon.png")
            img = img.resize((400, 400), Image.LANCZOS)
            self.logo = ImageTk.PhotoImage(img)
            logo_label = Label(right_frame, image=self.logo, bg=LIGHT_GRAY)
            logo_label.pack()
        except:
            canvas = Canvas(right_frame, width=400, height=400, bg=LIGHT_GRAY, highlightthickness=0)
            canvas.pack()
            canvas.create_text(200, 200, text="Healthcare\nAssistant", 
                             font=('Helvetica', 24, 'bold'), fill=DARK_GRAY)
        
        # Buttons frame - made more prominent
        btn_frame = Frame(content_frame, bg=LIGHT_GRAY)
        btn_frame.pack(pady=30)
        
        Button(btn_frame, text="LOGIN", 
              command=lambda: controller.show_frame("LoginPage"), 
              font=controller.button_font,
              bg=ACCENT, fg=WHITE,
              padx=20, pady=10,
              relief=FLAT).grid(row=0, column=0, padx=10, pady=10)
        
        Button(btn_frame, text="REGISTER", 
              command=lambda: controller.show_frame("RegisterPage"),
              font=controller.button_font,
              bg=ACCENT, fg=WHITE,
              padx=20, pady=10,
              relief=FLAT).grid(row=0, column=1, padx=10, pady=10)
        
        # Guest login button
        Button(btn_frame, text="CONTINUE AS GUEST", 
              command=lambda: controller.show_frame("ChatbotPage"),
              font=controller.button_font,
              bg=SECONDARY, fg=WHITE,
              padx=20, pady=10,
              relief=FLAT).grid(row=1, column=0, columnspan=2, pady=10)
        
        # Team section
        team_frame = LabelFrame(content_frame, text=" Developed By ", font=controller.body_font,
                              bg=LIGHT_GRAY, padx=10, pady=10)
        team_frame.pack(fill=X, pady=10)
        
        members = ["Prince Kumar", "Ravinder", "Sneha Kukreti"]
        for i, member in enumerate(members):
            Label(team_frame, text=member, font=controller.body_font, bg=LIGHT_GRAY).grid(row=i, column=0, sticky=W, padx=10, pady=5)

class LoginPage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg=LIGHT_GRAY)
        self.controller = controller
        
        # Content frame
        content_frame = Frame(self, bg=LIGHT_GRAY)
        content_frame.place(relx=0.5, rely=0.5, anchor=CENTER, width=600, height=500)
        
        # Header
        header_frame = Frame(content_frame, bg=PRIMARY)
        header_frame.pack(fill=X, pady=(0, 30))
        
        Label(header_frame, text="Login to Your Account", 
             font=self.controller.title_font, bg=PRIMARY, fg=WHITE, padx=20, pady=10).pack(fill=X)
        
        # Form frame
        form_frame = Frame(content_frame, bg=LIGHT_GRAY, padx=20, pady=20)
        form_frame.pack(expand=True, fill=BOTH)
        
        # Username field
        Label(form_frame, text="Username:", font=self.controller.body_font, bg=LIGHT_GRAY).grid(row=0, column=0, sticky=W, pady=(20, 5))
        self.username_entry = Entry(form_frame, font=self.controller.body_font)
        self.username_entry.grid(row=0, column=1, pady=(20, 5), padx=10, sticky=EW)
        
        # Password field
        Label(form_frame, text="Password:", font=self.controller.body_font, bg=LIGHT_GRAY).grid(row=1, column=0, sticky=W, pady=10)
        self.password_entry = Entry(form_frame, show="•", font=self.controller.body_font)
        self.password_entry.grid(row=1, column=1, pady=10, padx=10, sticky=EW)
        
        # Remember me checkbox
        self.remember_var = IntVar()
        Checkbutton(form_frame, text="Remember me", variable=self.remember_var, 
                   font=self.controller.body_font, bg=LIGHT_GRAY).grid(row=2, column=1, sticky=W, pady=10)
        
        # Buttons frame
        btn_frame = Frame(form_frame, bg=LIGHT_GRAY)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        Button(btn_frame, text="Login", 
              command=self.authenticate, 
              font=self.controller.button_font,
              bg=ACCENT, fg=WHITE,
              padx=20, pady=10,
              relief=FLAT).pack(side=LEFT, padx=10)
        
        Button(btn_frame, text="Back", 
              command=lambda: controller.show_frame("MainPage"),
              font=self.controller.button_font,
              bg=SECONDARY, fg=WHITE,
              padx=20, pady=10,
              relief=FLAT).pack(side=LEFT, padx=10)
        
        # Configure grid weights
        form_frame.columnconfigure(1, weight=1)
        
        # Bind Enter key to login
        self.password_entry.bind('<Return>', lambda event: self.authenticate())
    
    def authenticate(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        
        if self.controller.authenticate(username, password):
            self.controller.show_frame("ChatbotPage")

class RegisterPage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg=LIGHT_GRAY)
        self.controller = controller
        
        # Content frame
        content_frame = Frame(self, bg=LIGHT_GRAY)
        content_frame.place(relx=0.5, rely=0.5, anchor=CENTER, width=600, height=500)
        
        # Header
        header_frame = Frame(content_frame, bg=PRIMARY)
        header_frame.pack(fill=X, pady=(0, 30))
        
        Label(header_frame, text="Create New Account", 
             font=self.controller.title_font, bg=PRIMARY, fg=WHITE, padx=20, pady=10).pack(fill=X)
        
        # Form frame
        form_frame = Frame(content_frame, bg=LIGHT_GRAY, padx=20, pady=20)
        form_frame.pack(expand=True, fill=BOTH)
        
        # Username field
        Label(form_frame, text="Username:", font=self.controller.body_font, bg=LIGHT_GRAY).grid(row=0, column=0, sticky=W, pady=(20, 5))
        self.username_entry = Entry(form_frame, font=self.controller.body_font)
        self.username_entry.grid(row=0, column=1, pady=(20, 5), padx=10, sticky=EW)
        
        # Password field
        Label(form_frame, text="Password:", font=self.controller.body_font, bg=LIGHT_GRAY).grid(row=1, column=0, sticky=W, pady=10)
        self.password_entry = Entry(form_frame, show="•", font=self.controller.body_font)
        self.password_entry.grid(row=1, column=1, pady=10, padx=10, sticky=EW)
        
        # Confirm Password field
        Label(form_frame, text="Confirm Password:", font=self.controller.body_font, bg=LIGHT_GRAY).grid(row=2, column=0, sticky=W, pady=10)
        self.confirm_pass_entry = Entry(form_frame, show="•", font=self.controller.body_font)
        self.confirm_pass_entry.grid(row=2, column=1, pady=10, padx=10, sticky=EW)
        
        # Buttons frame
        btn_frame = Frame(form_frame, bg=LIGHT_GRAY)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        Button(btn_frame, text="Register", 
              command=self.register, 
              font=self.controller.button_font,
              bg=ACCENT, fg=WHITE,
              padx=20, pady=10,
              relief=FLAT).pack(side=LEFT, padx=10)
        
        Button(btn_frame, text="Back", 
              command=lambda: controller.show_frame("MainPage"),
              font=self.controller.button_font,
              bg=SECONDARY, fg=WHITE,
              padx=20, pady=10,
              relief=FLAT).pack(side=LEFT, padx=10)
        
        # Configure grid weights
        form_frame.columnconfigure(1, weight=1)
        
        # Bind Enter key to register
        self.confirm_pass_entry.bind('<Return>', lambda event: self.register())
    
    def register(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        confirm_password = self.confirm_pass_entry.get()
        
        if password != confirm_password:
            self.controller.update_status("Passwords do not match")
            return
            
        if self.controller.register(username, password):
            self.controller.show_frame("ChatbotPage")

class ChatbotPage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg=LIGHT_GRAY)
        self.controller = controller
        self.user_symptoms = []
        
        # Main frame
        main_frame = Frame(self, bg=LIGHT_GRAY)
        main_frame.pack(expand=True, fill=BOTH, padx=20, pady=20)
        
        # Header
        header_frame = Frame(main_frame, bg=PRIMARY)
        header_frame.pack(fill=X, pady=(0, 20))
        
        Label(header_frame, text="Healthcare Diagnostic Assistant", 
             font=self.controller.title_font, bg=PRIMARY, fg=WHITE, padx=20, pady=10).pack(fill=X)
        
        # Chat area
        chat_frame = Frame(main_frame, bg=LIGHT_GRAY)
        chat_frame.pack(fill=BOTH, expand=True)
        
        # Welcome message
        welcome_frame = Frame(chat_frame, bg=LIGHT_GRAY, padx=10, pady=10)
        welcome_frame.pack(fill=X, pady=5)
        
        welcome_msg = ("Welcome to the Healthcare Assistant! Start by entering your symptoms. "
                      "The system will analyze them and provide a preliminary diagnosis.")
        
        welcome_label = Label(welcome_frame, text=welcome_msg, wraplength=800,
                            font=self.controller.body_font, justify=LEFT, 
                            bg=LIGHT_GRAY, padx=10, pady=10)
        welcome_label.pack(fill=X)
        
        # Symptom input area
        input_frame = LabelFrame(chat_frame, text=" Enter Your Symptoms ", 
                               font=self.controller.body_font,
                               bg=LIGHT_GRAY, padx=10, pady=10)
        input_frame.pack(fill=X, pady=5)
        
        self.symptom_entry = Entry(input_frame, font=self.controller.body_font)
        self.symptom_entry.pack(side=LEFT, fill=X, expand=True, padx=5)
        
        Button(input_frame, text="Add Symptom", command=self.add_symptom,
              font=self.controller.button_font,
              bg=ACCENT, fg=WHITE,
              padx=10, pady=5,
              relief=FLAT).pack(side=LEFT, padx=5)
        
        # Bind Enter key to add symptom
        self.symptom_entry.bind('<Return>', lambda event: self.add_symptom())
        
        # Selected symptoms display
        selected_frame = LabelFrame(chat_frame, text=" Selected Symptoms ", 
                                  font=self.controller.body_font,
                                  bg=LIGHT_GRAY, padx=10, pady=10)
        selected_frame.pack(fill=X, pady=5)
        
        self.selected_symptoms_text = scrolledtext.ScrolledText(selected_frame, 
                                                             height=4, 
                                                             wrap=WORD,
                                                             font=self.controller.body_font,
                                                             padx=10,
                                                             pady=10)
        self.selected_symptoms_text.pack(fill=BOTH, expand=True)
        
        # Diagnosis area
        diagnosis_frame = LabelFrame(chat_frame, text=" Diagnosis Results ", 
                                   font=self.controller.body_font,
                                   bg=LIGHT_GRAY, padx=10, pady=10)
        diagnosis_frame.pack(fill=BOTH, expand=True, pady=5)
        
        self.diagnosis_text = scrolledtext.ScrolledText(diagnosis_frame, 
                                                      height=15, 
                                                      wrap=WORD,
                                                      font=self.controller.body_font,
                                                      padx=10,
                                                      pady=10)
        self.diagnosis_text.pack(fill=BOTH, expand=True)
        
        # Configure text tags for styling
        self.diagnosis_text.tag_configure("bold", font=('Helvetica', 14, 'bold'))
        self.diagnosis_text.tag_configure("accent", foreground=ACCENT)
        
        # Button controls
        control_frame = Frame(main_frame, bg=LIGHT_GRAY)
        control_frame.pack(fill=X, pady=10)
        
        Button(control_frame, text="Analyze Symptoms", 
              command=self.analyze_symptoms,
              font=self.controller.button_font,
              bg=SUCCESS, fg=WHITE,
              padx=10, pady=5,
              relief=FLAT).pack(side=LEFT, padx=5)
        
        Button(control_frame, text="Clear All", 
              command=self.clear_symptoms,
              font=self.controller.button_font,
              bg=WARNING, fg=WHITE,
              padx=10, pady=5,
              relief=FLAT).pack(side=LEFT, padx=5)
        
        Button(control_frame, text="Traditional Diagnosis", 
              command=lambda: controller.show_frame("TraditionalDiagnosisPage"),
              font=self.controller.button_font,
              bg=ACCENT, fg=WHITE,
              padx=10, pady=5,
              relief=FLAT).pack(side=LEFT, padx=5)
        
        Button(control_frame, text="Logout", 
              command=lambda: controller.show_frame("MainPage"),
              font=self.controller.button_font,
              bg=ERROR, fg=WHITE,
              padx=10, pady=5,
              relief=FLAT).pack(side=RIGHT, padx=5)
    
    def add_symptom(self):
        """Add symptom to the list after fuzzy matching"""
        symptom = self.symptom_entry.get().strip()
        if not symptom:
            return
            
        # Find closest match in our symptom list
        matches = get_close_matches(symptom, self.controller.all_symptoms, n=1, cutoff=0.6)
        
        if matches:
            matched_symptom = matches[0]
            if matched_symptom not in self.user_symptoms:
                self.user_symptoms.append(matched_symptom)
                self.selected_symptoms_text.insert(END, f"• {matched_symptom}\n")
                self.symptom_entry.delete(0, END)
                self.controller.update_status(f"Added symptom: {matched_symptom}")
            else:
                self.controller.update_status("Symptom already added")
        else:
            self.controller.update_status("No matching symptom found. Please try different wording.")
    
    def analyze_symptoms(self):
        """Analyze the entered symptoms and provide diagnosis"""
        if not self.user_symptoms:
            self.controller.update_status("Please add at least one symptom")
            return
            
        self.diagnosis_text.delete(1.0, END)
        
        # Show loading message
        self.diagnosis_text.insert(END, "Analyzing symptoms...\n\n")
        self.diagnosis_text.update()
        
        # Get analysis from controller
        result = self.controller.analyze_symptoms(self.user_symptoms)
        
        if not result:
            return
            
        # Display results
        self.diagnosis_text.insert(END, "You may have: ", "bold")
        self.diagnosis_text.insert(END, f"{result['disease']}\n\n", "accent")
        
        self.diagnosis_text.insert(END, "Symptoms you reported:\n", "bold")
        for symptom in result['symptoms_present']:
            self.diagnosis_text.insert(END, f"• {symptom}\n")
        
        self.diagnosis_text.insert(END, "\nCommon symptoms of this condition:\n", "bold")
        for symptom in result['symptoms_given']:
            self.diagnosis_text.insert(END, f"• {symptom}\n")
        
        self.diagnosis_text.insert(END, "\nConfidence level: ", "bold")
        self.diagnosis_text.insert(END, f"{result['confidence']:.1%}\n\n", "accent")
        
        # Doctor recommendation
        if result['doctor']:
            self.diagnosis_text.insert(END, "Recommended specialist:\n", "bold")
            self.diagnosis_text.insert(END, f"• {result['doctor']}\n\n")
            
            if result['doctor_link']:
                hyperlink = HyperlinkManager(self.diagnosis_text)
                def click1():
                    webbrowser.open_new(str(result['doctor_link']))
                self.diagnosis_text.insert(END, "More information: ", "bold")
                self.diagnosis_text.insert(END, "Visit ", hyperlink.add(click1))
                self.diagnosis_text.insert(END, str(result['doctor_link']) + "\n")
    
    def clear_symptoms(self):
        """Clear all entered symptoms"""
        self.user_symptoms = []
        self.selected_symptoms_text.delete(1.0, END)
        self.diagnosis_text.delete(1.0, END)
        self.controller.update_status("Cleared all symptoms")

class TraditionalDiagnosisPage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg=LIGHT_GRAY)
        self.controller = controller
        self.symptoms_present = []
        self.current_node = 0
        
        # Main frame
        main_frame = Frame(self, bg=LIGHT_GRAY)
        main_frame.pack(expand=True, fill=BOTH, padx=20, pady=20)
        
        # Header
        header_frame = Frame(main_frame, bg=PRIMARY)
        header_frame.pack(fill=X, pady=(0, 20))
        
        Label(header_frame, text="Traditional Diagnosis", 
             font=self.controller.title_font, bg=PRIMARY, fg=WHITE, padx=20, pady=10).pack(fill=X)
        
        # Question area
        question_frame = LabelFrame(main_frame, text=" Question ", 
                                  font=self.controller.body_font,
                                  bg=LIGHT_GRAY, padx=10, pady=10)
        question_frame.pack(fill=X, pady=5)
        
        self.question_text = Text(question_frame, 
                                height=4, 
                                wrap=WORD, 
                                font=self.controller.body_font,
                                padx=10,
                                pady=10)
        self.question_text.pack(fill=BOTH, expand=True)
        
        # Response area
        response_frame = LabelFrame(main_frame, text=" Diagnosis ", 
                                  font=self.controller.body_font,
                                  bg=LIGHT_GRAY, padx=10, pady=10)
        response_frame.pack(fill=BOTH, expand=True, pady=5)
        
        self.response_text = scrolledtext.ScrolledText(response_frame, 
                                                     height=15, 
                                                     wrap=WORD,
                                                     font=self.controller.body_font,
                                                     padx=10,
                                                     pady=10)
        self.response_text.pack(fill=BOTH, expand=True)
        
        # Configure text tags for styling
        self.response_text.tag_configure("bold", font=('Helvetica', 14, 'bold'))
        self.response_text.tag_configure("accent", foreground=ACCENT)
        
        # Button controls
        control_frame = Frame(main_frame, bg=LIGHT_GRAY)
        control_frame.pack(fill=X, pady=10)
        
        Button(control_frame, text="Yes", 
              command=self.answer_yes,
              font=self.controller.button_font,
              bg=SUCCESS, fg=WHITE,
              padx=10, pady=5,
              relief=FLAT).pack(side=LEFT, padx=5)
        
        Button(control_frame, text="No", 
              command=self.answer_no,
              font=self.controller.button_font,
              bg=ERROR, fg=WHITE,
              padx=10, pady=5,
              relief=FLAT).pack(side=LEFT, padx=5)
        
        Button(control_frame, text="Clear", 
              command=self.clear_response,
              font=self.controller.button_font,
              bg=WARNING, fg=WHITE,
              padx=10, pady=5,
              relief=FLAT).pack(side=LEFT, padx=5)
        
        Button(control_frame, text="Back", 
              command=lambda: controller.show_frame("ChatbotPage"),
              font=self.controller.button_font,
              bg=SECONDARY, fg=WHITE,
              padx=10, pady=5,
              relief=FLAT).pack(side=RIGHT, padx=5)
        
        # Start the diagnosis
        self.ask_question()
    
    def ask_question(self):
        """Ask the next question in the decision tree"""
        if self.controller.classifier.tree_.feature[self.current_node] != _tree.TREE_UNDEFINED:
            question = self.controller.cols[self.controller.classifier.tree_.feature[self.current_node]] + "?"
            self.question_text.delete(1.0, END)
            self.question_text.insert(END, question)
        else:
            self.provide_diagnosis()
    
    def answer_yes(self):
        """Process yes answer"""
        self.symptoms_present.append(self.controller.cols[self.controller.classifier.tree_.feature[self.current_node]])
        self.current_node = self.controller.classifier.tree_.children_right[self.current_node]
        self.ask_question()
    
    def answer_no(self):
        """Process no answer"""
        self.current_node = self.controller.classifier.tree_.children_left[self.current_node]
        self.ask_question()
    
    def provide_diagnosis(self):
        """Provide final diagnosis in traditional format"""
        try:
            # Get analysis from controller
            result = self.controller.analyze_symptoms(self.symptoms_present)
            
            if not result:
                return
                
            self.response_text.delete(1.0, END)
            
            # Display results
            self.response_text.insert(END, "You may have: ", "bold")
            self.response_text.insert(END, f"{result['disease']}\n\n", "accent")
            
            self.response_text.insert(END, "Symptoms you reported:\n", "bold")
            for symptom in result['symptoms_present']:
                self.response_text.insert(END, f"• {symptom}\n")
            
            self.response_text.insert(END, "\nCommon symptoms of this condition:\n", "bold")
            for symptom in result['symptoms_given']:
                self.response_text.insert(END, f"• {symptom}\n")
            
            self.response_text.insert(END, "\nConfidence level: ", "bold")
            self.response_text.insert(END, f"{result['confidence']:.1%}\n\n", "accent")
            
            # Doctor recommendation
            if result['doctor']:
                self.response_text.insert(END, "Recommended specialist:\n", "bold")
                self.response_text.insert(END, f"• {result['doctor']}\n\n")
                
                if result['doctor_link']:
                    hyperlink = HyperlinkManager(self.response_text)
                    def click1():
                        webbrowser.open_new(str(result['doctor_link']))
                    self.response_text.insert(END, "More information: ", "bold")
                    self.response_text.insert(END, "Visit ", hyperlink.add(click1))
                    self.response_text.insert(END, str(result['doctor_link']) + "\n")
            
        except Exception as e:
            self.controller.update_status(f"Error during diagnosis: {str(e)}")
            self.response_text.insert(END, f"Error: {str(e)}\n")
    
    def clear_response(self):
        """Clear the response area"""
        self.response_text.delete(1.0, END)
        self.controller.update_status("Cleared diagnosis results")

if __name__ == "__main__":
    app = HealthcareChatbot()
    app.run()