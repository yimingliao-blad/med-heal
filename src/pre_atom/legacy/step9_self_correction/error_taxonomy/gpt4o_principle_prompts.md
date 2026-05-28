# GPT-4o Principle-Based Detection Prompts

### Prompt 1: Structured Analysis

**Objective:** Decompose the question and methodically verify the answer against the discharge notes.

1. **Decompose the Question:**
   - Identify and note the specific hospital visit, clinical aspect, and time period mentioned in the question: {question}.
   - Clarify the key focus required for the answer.

2. **Alignment Check:**
   - Evaluate if the answer {answer} directly addresses the specific visit, aspect, and time period identified from the question.
   - Ensure the response is aligned with what was asked.

3. **Evidence Verification:**
   - Cross-examine each factual claim in the answer against the discharge notes {note}.
   - Look for contradictions or unsupported statements.

4. **Impact Assessment:**
   - Determine if any detected discrepancies or omissions would change the overall conclusion of the answer.
   - Focus on the significance of each detail in influencing the final response.

### Prompt 2: Logical Flow Evaluation

**Objective:** Use a logical sequence to verify the integrity and relevance of the answer.

1. **Question Breakdown:**
   - Parse the question {question} to extract the specific visit, clinical focus, and relevant time frame.
   - Establish the direct requirements for the answer.

2. **Relevance Check:**
   - Assess if the answer {answer} is appropriately tied to the identified visit and clinical context.
   - Confirm that the response stays true to the question's demands.

3. **Factual Consistency:**
   - Systematically verify each factual element of the answer with the discharge notes {note}.
   - Identify any inconsistencies or fabrications.

4. **Critical Error Analysis:**
   - Evaluate whether any discrepancies are crucial enough to alter the conclusion.
   - Focus only on errors that would lead to a different final answer.

### Prompt 3: Contextual Integrity Assessment

**Objective:** Ensure the answer maintains contextual integrity aligned with the question and supported by evidence.

1. **Question Contextualization:**
   - Dissect the question {question} to pinpoint the specific visit, aspect, and time frame required.
   - Frame the answer's scope accordingly.

2. **Alignment Verification:**
   - Scrutinize the answer {answer} for alignment with the context and specifics of the question.
   - Ensure that the response correlates with the intended query.

3. **Evidence-Based Scrutiny:**
   - Match each factual assertion in the answer to the discharge notes {note}.
   - Detect any unsupported or contradicted claims.

4. **Conclusion Integrity Check:**
   - Analyze if any identified errors are significant enough to impact the conclusion.
   - Distinguish between critical and minor errors, focusing on those affecting the final judgment.

Each prompt is designed to enforce the three principles in a distinct style, ensuring comprehensive self-critique while minimizing false positives and capturing misalignments effectively.