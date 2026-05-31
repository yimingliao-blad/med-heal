# GPT-4o Contradiction-Focused Detection Prompts

### Prompt 1: Focus on Direct Comparison

**Prompt Text:**
```
Review the following discharge note, question, and your own answer. Your task is to identify any factual contradictions between your answer and the discharge note. A contradiction occurs when your answer states a fact that directly conflicts with information in the note. 

1. **Extract** each factual claim from your answer.
2. **Locate** the relevant passage in the note for each claim.
3. **Compare** the claim with the note:
   - If the claim and the note provide conflicting information, describe the contradiction.
   - If the claim is unsupported by the note but does not conflict, do not flag it as an error.
   - Only flag omissions if the missing information would directly change your conclusion.

{note}

Question: {question}

Your Answer: {answer}
```

**Why It Works:**
This prompt explicitly guides the model to perform a step-by-step comparison for each factual claim, emphasizing the identification of conflicts rather than omissions. By focusing on "contradictions" and "conflicts," it steers the model away from defaulting to omissions.

**Difference from Others:**
This prompt is structured as a methodical process, explicitly breaking down the steps for comparison and emphasizing contradictions. It encourages the model to separate claims into discrete comparisons.

### Prompt 2: Emphasize Conflict Detection

**Prompt Text:**
```
You are tasked with ensuring your answer aligns with the discharge note and only identifying errors where specific contradictions exist. A contradiction means your answer provides information that directly disagrees with the note. 

1. **Identify** each factual statement in your answer.
2. **Search** for corresponding details in the note.
3. **Determine** if the details in the note contradict your statements:
   - Flag when your statement and the note provide opposing information.
   - If a detail is missing but does not alter the conclusion, it should not be flagged.

{note}

Question: {question}

Your Answer: {answer}
```

**Why It Works:**
This prompt stresses the importance of finding "opposing information," which narrows the focus to contradictions rather than omissions. It also clarifies that missing details should not be flagged unless they alter conclusions.

**Difference from Others:**
It uses language that highlights opposition and disagreement, which can help the model focus on finding true contradictions instead of just missing elements.

### Prompt 3: Contradiction-Centric Analysis

**Prompt Text:**
```
Analyze your answer in the context of the discharge note, focusing on identifying contradictions. A contradiction arises when your answer conflicts with the note. 

1. **Break down** your answer into factual claims.
2. **Cross-reference** each claim with the note.
3. **Identify** contradictions:
   - If a claim and the note show opposing facts, describe the issue.
   - Only consider omissions if they would change your answer significantly.

{note}

Question: {question}

Your Answer: {answer}
```

**Why It Works:**
This prompt encourages the model to "break down" claims and perform a "cross-reference," promoting thorough analysis for contradictions. The focus is on identifying "opposing facts," a clear directive for contradiction detection.

**Difference from Others:**
It emphasizes a more holistic "cross-reference" approach, encouraging the model to view the notes and answer in tandem rather than sequentially.

### Prompt 4: Logical Conflict Identification

**Prompt Text:**
```
Examine your answer against the discharge note to identify any logical conflicts. A conflict is present when your answer's statements disagree with the note's specifics. 

1. **List** factual elements of your answer.
2. **Match** each element with the note.
3. **Highlight** conflicts:
   - Flag instances where your statement and the note disagree.
   - Do not flag missing info unless it alters the answer's outcome.

{note}

Question: {question}

Your Answer: {answer}
```

**Why It Works:**
This prompt uses the term "logical conflicts," focusing the model on finding discrepancies that undermine the answer's validity. By instructing to "list" and "match," it encourages a systematic approach.

**Difference from Others:**
It uses a logical framing, suggesting that the notes and answer should logically align, thus naturally steering the model toward contradiction detection rather than superficial omissions.