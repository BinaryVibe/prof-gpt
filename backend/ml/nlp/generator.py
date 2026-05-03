import pandas as pd
import os

def generate_training_data():
    data = {
        "Query": [
            # Technical Concepts (Lectures)
            "What is the difference between supervised and unsupervised learning?",
            "Explain gradient descent.",
            "How does a Support Vector Machine work?",
            "What is the formula for Euclidean distance?",
            "Define the term backpropagation.",
            
            # Policy (Syllabus)
            "What is the penalty for late assignments?",
            "Is attendance mandatory for the labs?",
            "How much is the final exam worth?",
            "What is the plagiarism policy?",
            "Can we work in groups for the final project?",
            
            # Schedule (Calendar)
            "When is the midterm exam?",
            "What date is assignment 2 due?",
            "When are the professor's office hours?",
            "Is there class on Thanksgiving?",
            "What time does the Friday lab start?"
        ],
        "Intent": [
            "Technical", "Technical", "Technical", "Technical", "Technical",
            "Policy", "Policy", "Policy", "Policy", "Policy",
            "Schedule", "Schedule", "Schedule", "Schedule", "Schedule"
        ]
    }

    # Convert to a Pandas DataFrame
    df = pd.DataFrame(data)
    
    # Save it to a CSV file in the same directory
    output_path = os.path.join(os.path.dirname(__file__), "queries_dataset.csv")
    df.to_csv(output_path, index=False)
    
    print(f"✅ Successfully generated dataset with {len(df)} samples!")
    print(f"📁 Saved to: {output_path}")

if __name__ == "__main__":
    generate_training_data()