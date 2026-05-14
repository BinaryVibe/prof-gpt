from backend.database import client
from backend.ml.rag.embedder import miniLM_ef, ingest_chunks
from backend.ml.rag.chunker import split_document

def seed_evaluation_data():
    collection = client.get_or_create_collection(
        name="prof_gpt_materials", 
        embedding_function=miniLM_ef
    )

    print("🌱 Seeding the 10-Intent Knowledge Base...")

    dummy_files = [
        {"source": "lecture_slides.pdf", "category": "technical", "content": "React useEffect runs twice in StrictMode. KNN Manhattan distance is robust for sparse data. MongoDB $lookup requires precise local/foreign field mapping."},
        {"source": "grading_rubric.pdf", "category": "policy_handbook", "content": "The Terminal exam is worth 50% of the total marks. Sessional marks are distributed as 10, 15, and 25."},
        {"source": "student_handbook.pdf", "category": "policy_handbook", "content": "Minimum attendance is 80%. A medical certificate does not waive the attendance threshold, it only waives the short-attendance fine. If attendance is below 80% on CU Online, the portal's calculation is final regardless of manual tracking. Plagiarism results in a zero. A probation warning remains on the official transcript permanently, even if the student improves their CGPA above 2.0 in subsequent semesters."},
        {"source": "lab_guidelines.pdf", "category": "labs", "content": "Lab 3 OOP requires polymorphism implementation. Post-lab tasks must be submitted via the CU Online portal."},
        {"source": "cdc_internships.pdf", "category": "career_resources", "content": "Contact Sir Mushtaq Ahmed Bhatti for resume reviews. The Shine internship requires a minimum 3.0 CGPA."},
        {"source": "academic_calendar.pdf", "category": "admin_docs", "content": "Spring 2026 midterm week starts on April 15th."},
        {"source": "fee_structure.pdf", "category": "admin_docs", "content": "Generate your fee challan on CU Online. Installments are allowed if applied for before the deadline."},
        {"source": "exam_archive.pdf", "category": "exam_archive", "content": "Fall 2024 Terminal paper for Data Structures focused on Trees. OOP Sessional 1 had 5 coding questions."},
        {"source": "cls_acm_events.pdf", "category": "campus_life", "content": "The next CLS Mushaira is on Friday. Register for the Graphics Competition via the society portal."},
        {"source": "acm_membership.pdf", "category": "campus_life", "content": "Become an ACM Wah Chapter member by submitting the form. CODEZAAR is scheduled for October."}
    ]

    for item in dummy_files:
        print(f"Ingesting {item['source']}...")
        chunks = split_document(item['content'])
        ingest_chunks(chunks, collection, source_name=item['source'], category=item['category'])

    print("Database successfully seeded with evaluation data!")

if __name__ == "__main__":
    seed_evaluation_data()