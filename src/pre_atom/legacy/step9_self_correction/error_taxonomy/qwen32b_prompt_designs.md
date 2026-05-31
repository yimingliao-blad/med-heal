# Detection Prompt Designs (Qwen3-32B Generated)

### TYPE A — Rule-by-Rule Prompts  
**Prompt 1: Answer Ties to Question (Catches QUESTION_MISALIGNMENT)**  
```  
[Principle: Answer must directly address the question's focus.]  
Given:  
- Question: {question}  
- Answer: {answer}  

Check if the answer addresses the correct aspect of the question (e.g., correct visit, time period, or clinical detail). Does it avoid irrelevant information?  

Output YES/NO and evidence from the answer:  
```  

**Prompt 2: Faithful to Evidence (Catches MISREADING + FABRICATION)**  
```  
[Principle: All claims must be explicitly supported by the note.]  
Given:  
- Note: {note}  
- Answer: {answer}  

Check if the answer contains claims not present in the note (fabrication) or misrepresents details (e.g., wrong dosage, mixed-up medications).  

Output YES/NO and evidence from the note/answer:  
```  

**Prompt 3: Covers Key Details (Catches OMISSION)**  
```  
[Principle: Must include critical information from the note.]  
Given:  
- Note: {note}  
- Answer: {answer}  

Check if the answer omits critical details (e.g., key medications, diagnoses, or procedures) that are explicitly stated in the note. Minor omissions (e.g., missing non-critical dates) are acceptable.  

Output YES/NO and evidence from the note/answer:  
```  

**Variations for TYPE A**  
- **Aggressive**: Add "Assume any ambiguity in the answer is an error."  
- **Conservative**: Add "Only flag omissions if they impact clinical decision-making."  

---

### TYPE B — CoT Single Prompt  
```  
[Step-by-Step Self-Critique]  
Given:  
- Note: {note}  
- Question: {question}  
- Answer: {answer}  

1. **Alignment Check**: Does the answer directly address the question’s focus (e.g., correct visit, time period)? If not, flag as QUESTION_MISALIGNMENT.  
2. **Evidence Check**: Are all claims in the answer explicitly supported by the note? If not, flag as MISREADING (if the note contains conflicting info) or FABRICATION (if the note lacks the claim).  
3. **Coverage Check**: Does the answer include critical details from the note (e.g., medications, diagnoses)? If not, flag as OMISSION.  

Final Verdict: List all detected errors (if any).  
```  

**Variations for TYPE B**  
- **Strict Order**: Force the model to evaluate principles in reverse order (coverage → evidence → alignment).  
- **Weighted Prioritization**: Add "Prioritize flagging FABRICATION over minor omissions."  

---

### TYPE C — Few-Shot Prompt  
```  
[Analyze Errors in Medical Answers]  
Here are examples of how to detect errors in answers:  

**Example 1**  
- Question: What was the patient’s discharge medication for hypertension?  
- Note: "Discharged on Lisinopril 10mg daily. Metoprolol was held due to bradycardia."  
- Answer: "The patient was discharged on Metoprolol 50mg daily."  
- Error: **MISREADING** (Metoprolol was held; Lisinopril is correct).  

**Example 2**  
- Question: What were the discharge instructions for wound care?  
- Note: "Clean the incision twice daily with saline. Avoid strenuous activity for 2 weeks."  
- Answer: "The patient was instructed to clean the wound with hydrogen peroxide and return in 1 week."  
- Error: **FABRICATION** (Hydrogen peroxide not mentioned; return time unspecified).  

Now analyze this new case:  
- Note: {note}  
- Question: {question}  
- Answer: {answer}  
- Error:  
```  

**Variations for TYPE C**  
- **Error-Specific Examples**: Use one example for each error type (e.g., one for OMISSION, one for QUESTION_MISALIGNMENT).  
- **Contrastive Examples**: Include one correct answer and one wrong answer side-by-side.  

---

### Implementation Notes  
1. **For All Types**: Use `YES/NO` or `Error Type` outputs to simplify parsing.  
2. **Thresholding**: For TYPE A/B, aggregate errors across prompts (e.g., ≥1 error = reject answer).  
3. **Efficiency**: TYPE A is cheaper (3 small calls) but less holistic; TYPE B/C is more accurate but heavier.  

Let me know if you need code templates for integration!