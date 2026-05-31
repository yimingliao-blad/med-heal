# GPT-4o Creative Prompt Ideas (temp=1)

Creating a self-critique system that effectively detects and corrects its own errors requires innovative strategies to mitigate self-confirmation bias and enhance error detection. Here are five creative prompt strategies:

1. **Alternative Perspective Simulation**  
   **Explanation:** Encourage the model to adopt an alternative reading strategy by having it impersonate a third-party reviewer who examines the answer with a fresh perspective. Ask it to articulate possible misunderstandings that could arise from a quick or biased reading.  
   **Prompt Text:**  
   ```
   Pretend you are a reviewing clinician unfamiliar with the original analysis. First, rewrite the {note} in your own words to ensure a fresh interpretation. Now, considering the {question}, provide a verdict on the {answer} with potential misunderstandings highlighted. If any component seems potentially flawed, specify the error type and related content.  
   
   Reviewer's Analysis:
   - Rewritten Note: ...
   - Verdict: {"verdict": "CORRECT"/"INCORRECT", "error_type": "...", "wrong_claim": "...", "notes_say": "..."}
   ```  
   **Target Error Types:** MISREADING, QUESTION_MISALIGNMENT

2. **Reverse Inferencing**  
   **Explanation:** Guide the model to reverse-engineer the answer by considering a different starting point. Have it deduce what the {question} could be if the given {answer} is assumed correct, then check for discrepancies with the original {question}.  
   **Prompt Text:**  
   ```
   Imagine the {answer} is correct and generate possible {question} prompts that this {answer} would likely answer. Compare these generated questions with the actual {question}. Are there discrepancies that suggest an error? Provide a verdict with details.  
   
   Reverse Engineer Outcome:
   - Generated Questions: ...
   - Verdict: {"verdict": "CORRECT"/"INCORRECT", "error_type": "...", "wrong_claim": "...", "notes_say": "..."}
   ```  
   **Target Error Types:** QUESTION_MISALIGNMENT, FABRICATION

3. **Deconstructive Comparison**  
   **Explanation:** Ask the model to deconstruct the answer into its core claims and then compare each claim with components from the {note}. This granular analysis helps identify inaccuracies or unsupported claims.  
   **Prompt Text:**  
   ```
   Break down the {answer} into fundamental statements. For each statement, find supporting or conflicting information in the {note}. Highlight where data does not align, and formulate a verdict and error details if necessary.  
   
   Deconstructed Analysis:
   - Core Statements: ...
   - Verdict: {"verdict": "CORRECT"/"INCORRECT", "error_type": "...", "wrong_claim": "...", "notes_say": "..."}
   ```  
   **Target Error Types:** MISREADING, OMISSION, FABRICATION

4. **Paradoxical Challenge**  
   **Explanation:** Present a deliberate paradox by inserting an implausible error in the {note}, prompting the model to engage critically. This encourages a more active verification process to resolve contradictions.  
   **Prompt Text:**  
   ```
   The {note} contains an implausible error for this specific task. Identify the inconsistency involving the {answer}, given the intention to resolve this contradiction. Provide a verdict and describe the nature of any mistake found.
   
   Paradox Resolution:
   - Identified Inconsistency: ...
   - Verdict: {"verdict": "CORRECT"/"INCORRECT", "error_type": "...", "wrong_claim": "...", "notes_say": "..."}
   ```  
   **Target Error Types:** ALL

5. **Counterfactual Exploration**  
   **Explanation:** Invite the model to imagine a scenario where the answer is incorrect and explore what evidence would refute the {answer}. Doing so encourages deeper scrutiny and error acknowledgment.  
   **Prompt Text:**  
   ```
   Assume the {answer} is incorrect. Explore and describe what evidence from the {note} would decisively indicate a different conclusion. Compile your findings into a verdict and detailed error report.
   
   Counterfactual Analysis:
   - Refuting Evidence: ...
   - Verdict: {"verdict": "CORRECT"/"INCORRECT", "error_type": "...", "wrong_claim": "...", "notes_say": "..."}
   ```  
   **Target Error Types:** OMISSION, FABRICATION

These strategies are designed to leverage the model's capabilities while encouraging critical assessment, aiming for enhanced detection and selectivity beyond the previous best results.