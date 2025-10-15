import sqlite3

def get_evidence(db_path, page_title, sentence_id):
    """
    Quick evidence lookup from FEVER database
    
    Args:
        db_path: Path to fever.db
        page_title: Wikipedia page title (e.g., 'Doctor_Doom')
        sentence_id: Sentence number (0-indexed)
    
    Returns:
        Evidence sentence text
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query to get the sentence
    query = """
    SELECT lines 
    FROM documents 
    WHERE id = ?
    """
    
    cursor.execute(query, (page_title,))
    result = cursor.fetchone()
    
    if result:
        lines = result[0].split('\n')
        for line in lines:
            parts = line.split('\t')
            if len(parts) >= 2 and parts[0] == str(sentence_id):
                conn.close()
                return parts[1]  # Return the sentence text
    
    conn.close()
    return None

# Usage example
db_path = "/scratch/users/k24053411/fever_data/fever.db"

# Get Doctor Doom evidence (sentence 1)
evidence = get_evidence(db_path, "Doctor_Doom", 1)
print(f"Evidence: {evidence}")

# For your evidence format [[[193689, 204164, 'Doctor_Doom', 1]]]
def extract_evidence_from_refs(db_path, evidence_refs):
    """Extract evidence from FEVER reference format"""
    evidence_sentences = []
    
    for evidence_set in evidence_refs:
        for evidence_item in evidence_set:
            if len(evidence_item) >= 4:
                page_title = evidence_item[2]
                sentence_id = evidence_item[3]
                
                if page_title and sentence_id is not None:
                    sentence = get_evidence(db_path, page_title, sentence_id)
                    if sentence:
                        evidence_sentences.append(sentence)
    
    return evidence_sentences

# Test with your example
evidence_refs = [[[193689, 204164, 'Doctor_Doom', 1]]]
sentences = extract_evidence_from_refs(db_path, evidence_refs)
print(f"Extracted: {sentences}")