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
DARK_BLUE = "#0A2463"
LIGHT_BLUE = "#3E92CC"
WHITE = "#FFFAFF"
ACCENT = "#D8315B"
DARK_GRAY = "#1E1E1E"
LIGHT_GRAY = "#F5F5F5"

class HyperlinkManager:
    def __init__(self, text):
        self.text = text
        self.text.tag_config("hyper", foreground="blue", underline=1)
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
        self.root.geometry("1000x750")
        self.root.configure(bg=WHITE)
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('.', background=WHITE, foreground=DARK_GRAY)
        self.style.configure('TFrame', background=WHITE)
        self.style.configure('Header.TLabel', font=('Helvetica', 14, 'bold'), 
                           background=DARK_BLUE, foreground=WHITE, padding=10)
        self.style.configure('TButton', font=('Helvetica', 10), padding=5)
        self.style.configure('Accent.TButton', background=ACCENT, foreground=WHITE)
        self.style.map('Accent.TButton', 
                      background=[('active', ACCENT), ('!active', ACCENT)],
                      foreground=[('active', WHITE), ('!active', WHITE)])
        
        # Initialize pages
        self.create_main_page()
    
    def clear_window(self):
        """Clear all widgets from the window"""
        for widget in self.root.winfo_children():
            widget.destroy()
    
    def create_main_page(self):
        """Create the premium welcome page"""
        self.clear_window()
        
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(expand=True, fill=BOTH)
        
        # Header with gradient effect
        header_frame = ttk.Frame(main_frame, style='Header.TFrame')
        header_frame.pack(fill=X, pady=(0, 20))
        
        ttk.Label(header_frame, text="AI Healthcare Diagnostic Assistant", 
                 style='Header.TLabel').pack(expand=True)
        
        # Content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(expand=True, fill=BOTH)
        
        # Left side - description
        left_frame = ttk.Frame(content_frame, padding=10)
        left_frame.pack(side=LEFT, fill=BOTH, expand=True)
        
        ttk.Label(left_frame, text="Welcome to Your Personal", 
                 font=('Helvetica', 12)).pack(pady=5)
        ttk.Label(left_frame, text="Healthcare Assistant", 
                 font=('Helvetica', 16, 'bold'), foreground=ACCENT).pack(pady=5)
        
        ttk.Label(left_frame, 
                 text="Our AI-powered chatbot provides preliminary medical diagnosis\n"
                      "based on your symptoms and connects you with specialists.",
                 wraplength=300, justify=CENTER).pack(pady=20)
        
        # Right side - image placeholder
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=RIGHT, fill=BOTH, padx=20)
        
        try:
            img = Image.open("healthcare_icon.png")
            img = img.resize((300, 300), Image.LANCZOS)
            self.logo = ImageTk.PhotoImage(img)
            logo_label = Label(right_frame, image=self.logo, bg=WHITE)
            logo_label.pack()
        except:
            canvas = Canvas(right_frame, width=300, height=300, bg=LIGHT_GRAY, highlightthickness=0)
            canvas.pack()
            canvas.create_text(150, 150, text="Healthcare\nAssistant", 
                             font=('Helvetica', 18), fill=DARK_BLUE)
        
        # Buttons frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=20)
        
        login_btn = ttk.Button(btn_frame, text="LOGIN", 
                              command=self.show_login, 
                              style='Accent.TButton',
                              width=15)
        login_btn.grid(row=0, column=0, padx=10, pady=5)
        
        register_btn = ttk.Button(btn_frame, text="REGISTER", 
                                 command=self.show_register,
                                 style='Accent.TButton',
                                 width=15)
        register_btn.grid(row=0, column=1, padx=10, pady=5)
        
        # Team section
        team_frame = ttk.LabelFrame(main_frame, text=" Developed By ", padding=10)
        team_frame.pack(fill=X, pady=20)
        
        members = ["Prince Kumar", "Ravinder", "Sneha Kukreti"]
        for i, member in enumerate(members):
            ttk.Label(team_frame, text=member).grid(row=i, column=0, sticky=W, padx=10, pady=2)
    
    def show_login(self):
        """Show login form"""
        self.clear_window()
        
        login_frame = ttk.Frame(self.root, padding=20)
        login_frame.pack(expand=True)
        
        ttk.Label(login_frame, text="Login to Your Account", style='Header.TLabel').grid(row=0, column=0, columnspan=2, pady=10)
        
        ttk.Label(login_frame, text="Username:").grid(row=1, column=0, sticky=W, pady=5)
        self.username_entry = ttk.Entry(login_frame)
        self.username_entry.grid(row=1, column=1, pady=5, padx=5)
        
        ttk.Label(login_frame, text="Password:").grid(row=2, column=0, sticky=W, pady=5)
        self.password_entry = ttk.Entry(login_frame, show="*")
        self.password_entry.grid(row=2, column=1, pady=5, padx=5)
        
        btn_frame = ttk.Frame(login_frame)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=15)
        
        ttk.Button(btn_frame, text="Login", command=self.authenticate, style='Accent.TButton').pack(side=LEFT, padx=5)
        ttk.Button(btn_frame, text="Back", command=self.create_main_page).pack(side=LEFT, padx=5)
    
    def show_register(self):
        """Show registration form"""
        self.clear_window()
        
        reg_frame = ttk.Frame(self.root, padding=20)
        reg_frame.pack(expand=True)
        
        ttk.Label(reg_frame, text="Create New Account", style='Header.TLabel').grid(row=0, column=0, columnspan=2, pady=10)
        
        ttk.Label(reg_frame, text="Username:").grid(row=1, column=0, sticky=W, pady=5)
        self.reg_user_entry = ttk.Entry(reg_frame)
        self.reg_user_entry.grid(row=1, column=1, pady=5, padx=5)
        
        ttk.Label(reg_frame, text="Password:").grid(row=2, column=0, sticky=W, pady=5)
        self.reg_pass_entry = ttk.Entry(reg_frame, show="*")
        self.reg_pass_entry.grid(row=2, column=1, pady=5, padx=5)
        
        btn_frame = ttk.Frame(reg_frame)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=15)
        
        ttk.Button(btn_frame, text="Register", command=self.register, style='Accent.TButton').pack(side=LEFT, padx=5)
        ttk.Button(btn_frame, text="Back", command=self.create_main_page).pack(side=LEFT, padx=5)
    
    def authenticate(self):
        """Authenticate user login"""
        username = self.username_entry.get()
        password = self.password_entry.get()
        
        if not username or not password:
            messagebox.showerror("Error", "Please enter both username and password")
            return
            
        list_of_files = os.listdir()
        if username in list_of_files:
            with open(username, "r") as file:
                verify = file.read().splitlines()
                if password in verify:
                    messagebox.showinfo("Success", "Login Successful")
                    self.show_chatbot()
                else:
                    messagebox.showerror("Error", "Incorrect password")
        else:
            messagebox.showerror("Error", "User not found. Please register.")
    
    def register(self):
        """Register new user"""
        username = self.reg_user_entry.get()
        password = self.reg_pass_entry.get()
        
        if not username or not password:
            messagebox.showerror("Error", "Please enter both username and password")
            return
            
        if username in os.listdir():
            messagebox.showerror("Error", "Username already exists")
            return
            
        with open(username, "w") as file:
            file.write(f"{username}\n{password}")
        
        messagebox.showinfo("Success", "Registration successful")
        self.show_chatbot()
    
    def show_chatbot(self):
        """Show the premium chatbot interface"""
        self.clear_window()
        
        main_frame = ttk.Frame(self.root)
        main_frame.pack(expand=True, fill=BOTH, padx=20, pady=20)
        
        # Header
        header_frame = ttk.Frame(main_frame, style='Header.TFrame')
        header_frame.pack(fill=X, pady=(0, 20))
        
        ttk.Label(header_frame, text="Healthcare Diagnostic Assistant", 
                 style='Header.TLabel').pack()
        
        # Chat area
        chat_frame = ttk.Frame(main_frame)
        chat_frame.pack(fill=BOTH, expand=True)
        
        # Symptom input area
        input_frame = ttk.LabelFrame(chat_frame, text=" Enter Your Symptoms ", padding=10)
        input_frame.pack(fill=X, pady=5)
        
        self.symptom_entry = ttk.Entry(input_frame, font=('Helvetica', 10))
        self.symptom_entry.pack(side=LEFT, fill=X, expand=True, padx=5)
        
        ttk.Button(input_frame, text="Add Symptom", command=self.add_symptom,
                  style='Accent.TButton').pack(side=LEFT, padx=5)
        
        # Selected symptoms display
        selected_frame = ttk.LabelFrame(chat_frame, text=" Selected Symptoms ", padding=10)
        selected_frame.pack(fill=X, pady=5)
        
        self.selected_symptoms_text = scrolledtext.ScrolledText(selected_frame, 
                                                             height=4, 
                                                             wrap=WORD,
                                                             font=('Helvetica', 10))
        self.selected_symptoms_text.pack(fill=BOTH, expand=True)
        
        # Diagnosis area
        diagnosis_frame = ttk.LabelFrame(chat_frame, text=" Diagnosis Results ", padding=10)
        diagnosis_frame.pack(fill=BOTH, expand=True, pady=5)
        
        self.diagnosis_text = scrolledtext.ScrolledText(diagnosis_frame, 
                                                      height=15, 
                                                      wrap=WORD,
                                                      font=('Helvetica', 10))
        self.diagnosis_text.pack(fill=BOTH, expand=True)
        
        # Button controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=X, pady=10)
        
        ttk.Button(control_frame, text="Analyze Symptoms", 
                  command=self.analyze_symptoms,
                  style='Accent.TButton').pack(side=LEFT, padx=5)
        
        ttk.Button(control_frame, text="Clear All", 
                  command=self.clear_symptoms).pack(side=LEFT, padx=5)
        
        ttk.Button(control_frame, text="Traditional Diagnosis", 
                  command=self.show_traditional_diagnosis,
                  style='Accent.TButton').pack(side=LEFT, padx=5)
        
        ttk.Button(control_frame, text="Logout", 
                  command=self.create_main_page).pack(side=RIGHT, padx=5)
        
        # Initialize symptom list
        self.user_symptoms = []
    
    def add_symptom(self):
        """Add symptom to the list after fuzzy matching"""
        symptom = self.symptom_entry.get().strip()
        if not symptom:
            return
            
        # Find closest match in our symptom list
        matches = get_close_matches(symptom, self.all_symptoms, n=1, cutoff=0.6)
        
        if matches:
            matched_symptom = matches[0]
            if matched_symptom not in self.user_symptoms:
                self.user_symptoms.append(matched_symptom)
                self.selected_symptoms_text.insert(END, f"- {matched_symptom}\n")
                self.symptom_entry.delete(0, END)
                messagebox.showinfo("Info", f"Added symptom: {matched_symptom}")
            else:
                messagebox.showinfo("Info", "Symptom already added")
        else:
            messagebox.showwarning("Warning", "No matching symptom found. Please try different wording.")
    
    def analyze_symptoms(self):
        """Analyze the entered symptoms and provide diagnosis"""
        if not self.user_symptoms:
            messagebox.showwarning("Warning", "Please add at least one symptom")
            return
            
        self.diagnosis_text.delete(1.0, END)
        
        try:
            # Convert symptoms to feature vector
            symptom_indices = [self.all_symptoms.index(symptom) for symptom in self.user_symptoms 
                            if symptom in self.all_symptoms]
            
            if not symptom_indices:
                self.diagnosis_text.insert(END, "No valid symptoms found for analysis")
                return
                
            # Create feature vector
            X = np.zeros(len(self.all_symptoms))
            for idx in symptom_indices:
                X[idx] = 1
                
            # Predict disease
            prediction = self.classifier.predict([X])
            disease = self.labelencoder.inverse_transform(prediction)[0]
            
            # Get disease info - exactly like original console version
            self.diagnosis_text.insert(END, "You may have " + str(disease) + "\n\n")
            
            # Get all symptoms for this disease - fixed implementation
            disease_data = self.dimensionality_reduction.loc[disease]
            symptoms_given = [col for col, val in zip(self.cols, disease_data) if val == 1]
            
            # Display symptoms present - exactly like original
            self.diagnosis_text.insert(END, "symptoms present  " + str(self.user_symptoms) + "\n\n")
            
            # Display all related symptoms - exactly like original
            self.diagnosis_text.insert(END, "symptoms given " + str(symptoms_given) + "\n\n")
            
            # Calculate confidence level - fixed implementation
            if len(symptoms_given) > 0:
                confidence_level = len(set(self.user_symptoms) & set(symptoms_given)) / len(symptoms_given)
                self.diagnosis_text.insert(END, "confidence level is " + str(confidence_level) + "\n\n")
            else:
                self.diagnosis_text.insert(END, "confidence level is 0.0\n\n")
            
            # Doctor recommendation - exactly like original
            self.diagnosis_text.insert(END, "The model suggests:\n\n")
            row = self.doctors[self.doctors['disease'] == disease]
            
            if not row.empty:
                self.diagnosis_text.insert(END, "Consult " + str(row['Name'].values[0]) + "\n\n")
                
                # Add clickable link
                hyperlink = HyperlinkManager(self.diagnosis_text)
                def click1():
                    webbrowser.open_new(str(row['Description'].values[0]))
                self.diagnosis_text.insert(END, "Visit ", hyperlink.add(click1))
                self.diagnosis_text.insert(END, str(row['Description'].values[0]) + "\n")
            else:
                self.diagnosis_text.insert(END, "No doctor recommendation available for this condition\n")
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during diagnosis: {str(e)}")
            self.diagnosis_text.insert(END, f"Error: {str(e)}\n")
            
    def show_traditional_diagnosis(self):
        """Show the traditional yes/no question diagnosis"""
        self.clear_window()
        
        main_frame = ttk.Frame(self.root)
        main_frame.pack(expand=True, fill=BOTH, padx=20, pady=20)
        
        # Header
        header_frame = ttk.Frame(main_frame, style='Header.TFrame')
        header_frame.pack(fill=X, pady=(0, 20))
        
        ttk.Label(header_frame, text="Traditional Diagnosis", 
                 style='Header.TLabel').pack()
        
        # Question area
        question_frame = ttk.LabelFrame(main_frame, text=" Question ", padding=10)
        question_frame.pack(fill=X, pady=5)
        
        self.question_text = Text(question_frame, height=4, wrap=WORD, font=('Helvetica', 10))
        self.question_text.pack(fill=BOTH, expand=True)
        
        # Response area
        response_frame = ttk.LabelFrame(main_frame, text=" Diagnosis ", padding=10)
        response_frame.pack(fill=BOTH, expand=True, pady=5)
        
        self.response_text = scrolledtext.ScrolledText(response_frame, 
                                                     height=15, 
                                                     wrap=WORD,
                                                     font=('Helvetica', 10))
        self.response_text.pack(fill=BOTH, expand=True)
        
        # Button controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=X, pady=10)
        
        ttk.Button(control_frame, text="Yes", command=self.answer_yes,
                  style='Accent.TButton').pack(side=LEFT, padx=5)
        ttk.Button(control_frame, text="No", command=self.answer_no).pack(side=LEFT, padx=5)
        ttk.Button(control_frame, text="Clear", command=self.clear_response).pack(side=LEFT, padx=5)
        ttk.Button(control_frame, text="Back", command=self.show_chatbot).pack(side=RIGHT, padx=5)
        
        # Initialize diagnosis
        self.symptoms_present = []
        self.current_node = 0
        self.ask_question()
    
    def ask_question(self):
        """Ask the next question in the decision tree"""
        if self.classifier.tree_.feature[self.current_node] != _tree.TREE_UNDEFINED:
            question = self.cols[self.classifier.tree_.feature[self.current_node]] + "?"
            self.question_text.delete(1.0, END)
            self.question_text.insert(END, question)
        else:
            self.provide_diagnosis()
    
    def answer_yes(self):
        """Process yes answer"""
        self.symptoms_present.append(self.cols[self.classifier.tree_.feature[self.current_node]])
        self.current_node = self.classifier.tree_.children_right[self.current_node]
        self.ask_question()
    
    def answer_no(self):
        """Process no answer"""
        self.current_node = self.classifier.tree_.children_left[self.current_node]
        self.ask_question()
    
    def print_disease(self, node):
        """Print disease from node value (matches original console version)"""
        node = node[0]
        val = node.nonzero() 
        disease = self.labelencoder.inverse_transform(val[0])
        return disease
    
    def provide_diagnosis(self):
        """Provide final diagnosis in traditional format"""
        try:
            present_disease = self.print_disease(self.classifier.tree_.value[self.current_node])
            
            self.response_text.delete(1.0, END)
            self.response_text.insert(END, "You may have " + str(present_disease) + "\n\n")
            
            # Get all symptoms for this disease - fixed implementation
            disease_data = self.dimensionality_reduction.loc[present_disease]
            symptoms_given = [col for col, val in zip(self.cols, disease_data) if val == 1]
            
            # Display symptoms present - exactly like original
            self.response_text.insert(END, "symptoms present  " + str(self.symptoms_present) + "\n\n")
            
            # Display all related symptoms - exactly like original
            self.response_text.insert(END, "symptoms given " + str(symptoms_given) + "\n\n")
            
            # Calculate confidence level - fixed implementation
            if len(symptoms_given) > 0:
                confidence_level = len(set(self.symptoms_present) & set(symptoms_given)) / len(symptoms_given)
                self.response_text.insert(END, "confidence level is " + str(confidence_level) + "\n\n")
            else:
                self.response_text.insert(END, "confidence level is 0.0\n\n")
            
            # Doctor recommendation - exactly like original
            self.response_text.insert(END, "The model suggests:\n\n")
            row = self.doctors[self.doctors['disease'] == present_disease[0]]
            
            if not row.empty:
                self.response_text.insert(END, "Consult " + str(row['Name'].values[0]) + "\n\n")
                
                # Add clickable link
                hyperlink = HyperlinkManager(self.response_text)
                def click1():
                    webbrowser.open_new(str(row['Description'].values[0]))
                self.response_text.insert(END, "Visit ", hyperlink.add(click1))
                self.response_text.insert(END, str(row['Description'].values[0]) + "\n")
            else:
                self.response_text.insert(END, "No doctor recommendation available for this condition\n")
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during diagnosis: {str(e)}")
            self.response_text.insert(END, f"Error: {str(e)}\n")
    
    def clear_response(self):
        """Clear the response area"""
        self.response_text.delete(1.0, END)
    
    def clear_symptoms(self):
        """Clear all entered symptoms"""
        self.user_symptoms = []
        self.selected_symptoms_text.delete(1.0, END)
        self.diagnosis_text.delete(1.0, END)
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = HealthcareChatbot()
    app.run()