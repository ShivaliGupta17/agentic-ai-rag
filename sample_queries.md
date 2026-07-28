

### 1. Definitional question

**Query:** `What is Agentic AI?`

**Response:**
```json
{
  "answer": "Agentic AI refers to systems capable of autonomous decision-making and action in pursuit of specific objectives. Unlike a passive tool, it understands context, breaks down complex goals, makes independent decisions, learns and adapts dynamically, and takes initiative without constant human supervision.",
  "context_chunks": [
    {"text": "Agentic AI refers to systems capable of autonomous decision-making...", "page": 12, "score": 0.78},
    {"text": "At its core, Agentic AI is about: Understanding context beyond literal instructions...", "page": 2, "score": 0.69}
  ],
  "confidence": 0.78
}
```

---

### 2. Comparison question

**Query:** `How is Agentic AI different from a regular LLM?`

**Response:**
```json
{
  "answer": "LLMs are reactive — they respond to prompts and generate text or language, but have no inherent decision-making capability. Agents go further: they are context-aware, goal-driven, operate continuously with minimal human intervention, and can integrate multiple tools (including LLMs) to actually take actions toward a goal.",
  "context_chunks": [
    {"text": "Aspect | LLMs | Agents ... Interactivity: respond to inputs (reactive) vs operate continuously...", "page": 4, "score": 0.74}
  ],
  "confidence": 0.74
}
```

---

### 3. List / enumeration question

**Query:** `What industries does the eBook give Agentic AI use cases for?`

**Response:**
```json
{
  "answer": "The eBook covers use cases across Retail, Manufacturing, Healthcare, Biosciences, Pharmaceuticals, Finance & Insurance, Education, Telecommunications, and Construction.",
  "context_chunks": [
    {"text": "1. Retail ... 2. Manufacturing ... 3. Healthcare ...", "page": 8, "score": 0.81}
  ],
  "confidence": 0.81
}
```

---

### 4. Numeric / statistic question

**Query:** `What productivity improvement did the retail company in the eBook's scenario see?`

**Response:**
```json
{
  "answer": "According to the retail scenario described, AI support systems gave employees 30% more time to focus on high-priority projects like marketing and product development.",
  "context_chunks": [
    {"text": "Employee Productivity: AI support systems have given employees 30% more time...", "page": 7, "score": 0.83}
  ],
  "confidence": 0.83
}
```

---

### 5. Multi-hop / structural question

**Query:** `What are the seven layers of a multi-agent system's architecture?`

**Response:**
```json
{
  "answer": "The eBook describes seven layers: Perception, Representation, Decision-Making, Planning, Action, Interaction, and Learning.",
  "context_chunks": [
    {"text": "Perception Layer: ... Representation Layer: ...", "page": 25, "score": 0.76},
    {"text": "Decision-Making Layer: ... Planning Layer: ...", "page": 26, "score": 0.71}
  ],
  "confidence": 0.76
}
```

---

### 6. Out-of-scope question — refusal case (proves grounding works)

**Query:** `What is the capital of France?`

**Response:**
```json
{
  "answer": "I couldn't find this in the document.",
  "context_chunks": [
    {"text": "Agentic AI is like having a super-intelligent collaborator...", "page": 2, "score": 0.11},
    {"text": "Foundational Agents Workflow Agents Utility Agents...", "page": 19, "score": 0.09}
  ],
  "confidence": 0.0
}
```

