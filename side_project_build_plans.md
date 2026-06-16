# Side Project Build Plans
*AI/ML Career Transition — ~14-Week Schedule*

---

## Master Timeline

| Week | Project 2: ML Pipeline | Project 1: AI Interview Coach | Project 3: Code/Test Analyzer |
|------|------------------------|-------------------------------|-------------------------------|
| 1 | Setup, EDA, problem framing | — | — |
| 2 | Feature engineering, data pipeline | — | — |
| 3 | Decision tree from scratch (DSA) | — | — |
| 4 | Model training, sklearn | Project setup, architecture | — |
| 5 | MLflow, hyperparameter tuning | FastAPI backend, LLM integration | — |
| 6 | FastAPI deployment, tests | Core interview flow, Pydantic | — |
| 7 | Monitoring, drift, polish & ship | Embeddings, semantic matching | — |
| 8 | — | Frontend: Next.js setup, skeleton | — |
| 9 | — | Frontend: charts, animations | — |
| 10 | — | Session persistence, dashboard | AST parsing, call graph |
| 11 | — | Polish, Vercel deploy & ship | Graph traversal algorithms |
| 12 | — | — | LLM integration, RAG setup |
| 13 | — | — | Observer pattern, plugin system |
| 14 | — | — | Polish, tests, README & ship |

*Projects overlap intentionally — shift timelines as needed. Week 15+ is your buffer.*

---

## Project 2: End-to-End ML Pipeline
**Weeks 1–7 | Goal: Build core ML competence with production-quality engineering discipline**

Start this one first. Getting ML fundamentals into your hands early means the concepts compound across all three projects.

Choose your dataset in week 1 — I recommend **customer churn prediction** (Telco dataset on Kaggle) because it's tabular (perfect for learning fundamentals), has a mix of numerical and categorical features, and the business problem is easy to explain to any interviewer.

---

### Week 1 — Environment, Data, and Problem Framing

**What to build:** A clean project skeleton and an exploratory data analysis (EDA) notebook.

Set up your repo with `pyproject.toml` (use `uv` or `poetry`), a `src/` layout, `pytest` configured from day one, and a `Makefile` with targets for `lint`, `test`, and `run`. The discipline of doing this upfront is part of the story — you're an SDET, your repo hygiene should be visibly better than the average ML candidate's.

In your EDA notebook, answer these questions about your data: What's the class balance? Which features have nulls and why? What's the distribution of each numerical feature? Which categorical features have high cardinality? Don't just run `.describe()` — write Markdown cells explaining what you observe and what it implies for modeling. This habit will serve you in technical interviews.

**DSA focus:** Implement a simple `FrequencyMap` class (a thin wrapper around a dict) for counting categorical value distributions. Not because you need it — because implementing small data structures from scratch gets your fingers remembering how Python works.

**Design pattern:** Set up a `DataLoader` base class using the **Template Method pattern**. Define the skeleton (`load → validate → preprocess → split`), leave the steps as abstract methods. You'll subclass it next week.

---

### Week 2 — Feature Engineering and the Data Pipeline

**What to build:** A reusable feature engineering pipeline and a concrete `TelcoDataLoader` subclass.

Implement your `TelcoDataLoader` extending last week's base class. For preprocessing: handle nulls with imputation (write your own mean/mode imputer first, then compare to sklearn's — the exercise matters more than the result), encode categoricals (one-hot and ordinal, understand the difference), and scale numericals (standard vs. min-max — know when each is appropriate).

Write unit tests for every transformer. Your SDET instincts are an asset here — test edge cases like empty columns, all-null columns, unseen categories at inference time. Most ML engineers skip this. You shouldn't.

**DSA focus:** Build a simple **LRU Cache** class to cache expensive preprocessing results during development. Use a `dict` + `collections.deque` — don't use `functools.lru_cache` yet. Understand the mechanics first.

**Design pattern:** Implement the **Pipeline pattern** — a `Pipeline` class that holds an ordered list of `Transformer` objects and calls `fit_transform` on each sequentially. This mirrors sklearn's Pipeline but you own the internals.

---

### Week 3 — Decision Tree from Scratch (The Core DSA Week)

**What to build:** A working `DecisionTreeClassifier` implemented from scratch, validated against sklearn's output.

This is the most important week of Project 2. A decision tree is a recursive data structure — building one from scratch cements your understanding of tree algorithms, recursion, and information theory all at once, and it's a legitimate interview talking point.

Your tree node: a `TreeNode` dataclass holding `feature_index`, `threshold`, `left`, `right`, `value` (for leaf nodes). Your training algorithm: recursive splitting using **information gain** (entropy of parent minus weighted entropy of children). Your stopping conditions: max depth, min samples per leaf, no information gain.

Implement `fit(X, y)` and `predict(X)` with clean type hints. Then write a test that trains both your implementation and `sklearn`'s on the same data and asserts their predictions match on a held-out set within a tolerance. If they diverge, debug until they don't.

**DSA focus:** This week *is* the DSA focus. You're implementing BFS traversal to print the tree, DFS recursion for training, and a depth-first search for inference.

**Design pattern:** Add a `Visitor` pattern so you can walk the trained tree and extract different views of it (depth, feature importances, prediction path for a single sample) without modifying the tree class itself.

---

### Week 4 — Model Training and Comparison

**What to build:** A multi-model training harness, cross-validation, and a results comparison table.

Now you get to use sklearn properly — train Logistic Regression, Random Forest, and Gradient Boosting alongside your scratch-built Decision Tree. Implement k-fold cross-validation manually (shuffle the data, slice into k folds, iterate) before using `sklearn.model_selection.cross_val_score` to verify.

Build a `ModelRegistry` — a dictionary mapping model names to fitted model instances — with a consistent `evaluate(X, y)` interface. Compute accuracy, precision, recall, F1, and AUC-ROC for each. Print a clean comparison table (use `rich` for nice terminal output).

**DSA focus:** Implement a **min-heap** (`heapq`) to track the top-k performing models by a given metric. Useful context: priority queues come up constantly in ML systems for things like beam search, candidate ranking, and job queues.

**Design pattern:** **Strategy pattern** for the evaluation metric. A `MetricStrategy` base class with concrete implementations for each metric. Your `ModelRegistry.evaluate()` accepts a strategy, making it trivial to swap what you're optimizing for.

---

### Week 5 — MLflow, Hyperparameter Tuning

**What to build:** Experiment tracking with MLflow and a hyperparameter search.

Install MLflow locally (`mlflow ui` spins up a local server at `localhost:5000`). Wrap your training loop so every run logs: parameters, metrics, the fitted model artifact, and a feature importance plot. After this week, you can show an interviewer a clean experiment history and explain what you tried and why.

Implement a simple **grid search** from scratch: nested loops over a parameter grid, training and evaluating each combination, logging each run to MLflow. Then implement **random search** (sample from parameter distributions) and compare — random search almost always finds better results faster. Understanding *why* is worth knowing.

**Design pattern:** **Observer pattern** for training callbacks. A `TrainingObserver` interface with `on_epoch_start`, `on_epoch_end`, `on_training_complete`. Concrete observers: `MLflowLogger`, `EarlyStopper`, `ProgressPrinter`. Your training loop calls `notify(observers)` at each hook. This mirrors how PyTorch Lightning, Keras, and sklearn callbacks work — you'll recognize the pattern everywhere.

---

### Week 6 — FastAPI Deployment and Tests

**What to build:** A REST API serving your best model, with full test coverage.

Serialize your best model with `joblib`. Build a FastAPI app with three endpoints: `POST /predict` (takes a JSON payload of feature values, returns prediction + probability), `GET /health` (model version, uptime, feature schema), and `GET /model-info` (feature importances, training metrics).

Use **Pydantic models** for input validation — define a `PredictionRequest` schema with field types and validators. Bad input should return a clear 422 with an explanation, not a 500. Write integration tests that spin up the API with `TestClient` and hit every endpoint. Your SDET background should make this section feel familiar — lean into it.

**Design pattern:** **Facade pattern** for the prediction service. Your API routes are thin; all the logic (load model, validate features, run inference, format output) lives in a `PredictionService` class. The API just delegates. Keeps routes clean and the service independently testable.

---

### Week 7 — Monitoring, Data Drift, Polish, and Ship

**What to build:** A data drift detector and a polished GitHub repo ready to show.

Data drift is one of the most important concepts in production ML — models degrade when the distribution of incoming data shifts away from training data. Implement a simple drift detector using the **Population Stability Index (PSI)** for numerical features and chi-squared test for categoricals. Log a warning when drift exceeds a threshold.

Add a `/drift-report` endpoint that compares a batch of recent predictions against the training distribution. Write a GitHub Actions workflow that runs your test suite on push. Write a proper `README.md` with: what the project does, architecture diagram (a simple ASCII one is fine), setup instructions, and example API calls with `curl`.

**Final checklist:** Clean commit history, no secrets in the repo, type hints throughout, docstrings on all public methods, test coverage above 80%.

---

## Project 1: AI Interview Coach
**Weeks 4–11 | Goal: Build the recruiter-facing portfolio centerpiece**

This starts in week 4 while you're still finishing the ML Pipeline. The two projects don't conflict — the ML Pipeline is heavily local/notebook work, while this is a web app. Starting the architecture thinking in week 4 means when you shift focus fully in week 8, you have momentum.

---

### Week 4 — Architecture Design and Project Setup

**What to build:** Repo scaffold, architecture diagram, state machine design.

Map out the full system before writing application code. The Interview Coach has three main components: a **backend** (FastAPI, Python), a **frontend** (Next.js + Tailwind + shadcn/ui), and **external services** (an LLM API, an embeddings API). Sketch the data flow: user submits JD → backend parses it → LLM generates questions → questions served to frontend → user submits answer → backend evaluates → scores returned.

Design your **state machine** on paper first. States: `IDLE`, `JD_LOADED`, `QUESTION_ACTIVE`, `ANSWER_SUBMITTED`, `EVALUATING`, `FEEDBACK_READY`, `SESSION_COMPLETE`. Transitions between them. Implement this as a `SessionStateMachine` class with explicit `transition(event)` method and guards that raise `InvalidTransitionError` if a transition isn't valid from the current state.

Set up two repos (or a monorepo with `apps/backend` and `apps/frontend`). Backend: FastAPI + SQLite (via SQLAlchemy for persistence) + `python-dotenv`. Frontend: `npx create-next-app` with Tailwind and shadcn/ui installed.

**Design pattern:** The state machine itself is a classic pattern. Also implement a **Registry** for question types — a dict mapping `QuestionType` enum values to `QuestionGenerator` classes, so adding a new type (e.g., system design questions) means adding one class and one registry entry, nothing else.

---

### Week 5 — FastAPI Backend and LLM Integration

**What to build:** Core backend endpoints and your first working LLM call.

Build these endpoints first: `POST /sessions` (create a new interview session, returns session ID), `POST /sessions/{id}/load-jd` (parse a job description, extract skills and role level), `GET /sessions/{id}/next-question` (return the next question based on session state).

For the LLM integration, use the **Strategy pattern** from the start — a `LLMProvider` abstract base class with `complete(prompt: str) -> str` and `embed(text: str) -> list[float]`. Concrete implementations for OpenAI and Anthropic. This means you can swap providers or test with a mock. Use `instructor` library (built on top of the OpenAI/Anthropic SDK) to get **structured output** — it forces the LLM to return a Pydantic model instead of free text. This is a production ML engineering skill worth highlighting.

Your JD parser should return a `JobDescription` Pydantic model: `role_title`, `seniority_level`, `required_skills: list[str]`, `nice_to_have_skills: list[str]`, `key_responsibilities: list[str]`. The LLM does the extraction; `instructor` enforces the schema.

**DSA focus:** Implement a **Trie** for skill tag autocomplete. When the user types in the skill filter on the frontend, the trie does prefix matching. Good excuse to implement one in Python cleanly.

---

### Week 6 — Interview Flow and Answer Submission

**What to build:** The complete question/answer/evaluate loop.

Add endpoints: `POST /sessions/{id}/submit-answer` (store the answer, trigger evaluation), `GET /sessions/{id}/evaluation/{answer_id}` (return evaluation results). 

Implement the **Chain of Responsibility** for answer evaluation. Each handler in the chain checks one dimension: `RelevanceChecker` (did they actually answer what was asked?), `DepthChecker` (was the answer substantive?), `CommunicationChecker` (was it clear and well-structured?), `TechnicalAccuracyChecker` (for technical questions — is it actually correct?). Each handler adds its score and notes to an `EvaluationContext` object and passes it to the next. Final output: a structured `EvaluationResult` with per-dimension scores, overall score, strengths, and improvement suggestions.

Build out your Pydantic models for everything: `Question`, `Answer`, `EvaluationResult`, `SessionSummary`. Good schema design here pays off in week 8 when the frontend consumes these.

---

### Week 7 — Embeddings and Semantic Matching

**What to build:** A semantic similarity layer that improves evaluation quality.

The chain-of-responsibility evaluator calls the LLM for each check, which is expensive. For the relevance check, use embeddings instead: embed the question, embed a set of "ideal answer themes" (generated once when the question is created), embed the user's answer, then compute cosine similarity. If similarity is below a threshold, the answer is off-topic without an LLM call.

Implement `cosine_similarity(a: list[float], b: list[float]) -> float` from scratch — it's a dot product divided by the product of magnitudes. Use numpy after implementing it manually. Store embeddings in SQLite as JSON-serialized float arrays (good enough for a portfolio project; mention you'd use pgvector or Pinecone in production).

**DSA focus:** Implement a **k-nearest neighbors** search over stored embeddings using a priority queue (`heapq.nlargest`). Given a new answer embedding, find the k most similar past answers the user has given. Use this to surface patterns: "You tend to give shallow answers on system design questions."

---

### Week 8 — Frontend: Next.js Skeleton and Core Flow

**What to build:** A working (if not yet beautiful) end-to-end flow in the browser.

Pages: `/` (landing/home), `/sessions/new` (JD upload), `/sessions/[id]/interview` (the interview flow), `/sessions/[id]/results` (post-session feedback), `/dashboard` (history across sessions).

For state management use **Zustand** — lightweight, and you won't need Redux for this scope. Your store holds: current session, current question, answer draft, evaluation result.

The interview page flow: JD textarea → "Start Interview" button → question card appears → user types answer in textarea → "Submit" → loading state → feedback panel slides in with scores and notes → "Next Question" → repeat. Make sure this actually works end-to-end with real API calls before moving to polish in week 9.

Use `fetch` with a thin API client module — a `createApiClient(baseUrl)` factory that returns typed functions for each endpoint. Avoid raw `fetch` calls scattered through components.

---

### Week 9 — Frontend: Polish, Charts, and Animations

**What to build:** The visual layer that makes this portfolio-shareable.

This week is purely about making it look impressive. Key UI elements:

**Radar chart** (Recharts `RadarChart`) on the results page — axes for Relevance, Depth, Communication, Technical Accuracy, Conciseness. Fills in as evaluations come back. This is the single most visually striking element.

**Question card** — a card component with a smooth `framer-motion` entrance animation, a countdown timer if you want to add time pressure (optional), and a subtle progress indicator showing question N of M.

**Feedback panel** — slides up from the bottom after submission. Color-coded score badges (green/yellow/red), expandable "Strengths" and "Suggestions" sections, a quote from the user's answer that exemplifies something good or needs improvement.

**Dashboard** — session history list with sparklines (Recharts `LineChart` with `syncId`) showing score trends over time per competency area. This is what you show in the first 30 seconds when a recruiter asks for the demo.

Use shadcn/ui components as your base — they look polished immediately and you'll spend time on layout rather than base component styling.

---

### Week 10 — Persistence, Session History, and Dashboard Data

**What to build:** SQLAlchemy models, session history, and the data layer feeding the dashboard.

Define your SQLAlchemy models: `InterviewSession`, `Question`, `Answer`, `Evaluation`. Write migrations with Alembic (good habit — shows you understand schema evolution). Add `GET /sessions` (paginated session history) and `GET /sessions/{id}/summary` (aggregate scores, question list, time spent).

Build the dashboard data endpoint: `GET /users/{id}/analytics` returning per-competency score trends over the last N sessions, most-asked question categories, average session length, improvement rate. The frontend consumes this to power the sparklines and summary cards.

Add background task processing with FastAPI's `BackgroundTasks` — evaluation should be kicked off as a background task so `POST /submit-answer` returns immediately with a `202 Accepted`, and the frontend polls or uses Server-Sent Events to get the result. This is how production APIs handle LLM latency.

---

### Week 11 — Polish, Testing, Vercel Deploy, and Ship

**What to build:** A publicly accessible URL you can put on your resume.

Write end-to-end tests using Playwright — automate the full happy path: load JD, answer three questions, view results. These are the kind of tests your SDET background makes you write better than anyone.

Deploy: backend to **Railway** or **Render** (free tier, zero config for FastAPI), frontend to **Vercel** (one command). Add a `DEMO_MODE` env flag that pre-loads a sample JD and uses canned LLM responses so the demo works without users hitting API rate limits.

Your `README.md` should include: a GIF of the demo (use Kap on Mac or ScreenToGif), architecture diagram, tech stack badges, and "How I built this" — a short paragraph explaining the design decisions. This is what recruiters read.

---

## Project 3: LLM-Powered Code & Test Analyzer
**Weeks 10–14 | Goal: Bridge your SDET background with AI in a compelling way**

This is your "career narrative" project. The story: *"I automated the code quality work I used to do manually by building an AI layer on top of static analysis."* Start this in week 10 while the Interview Coach is in its polish phase — the AST work is intellectually distinct enough that context-switching isn't painful.

---

### Week 10 — AST Parsing and Call Graph Construction

**What to build:** A Python AST parser that builds a directed call graph from a codebase.

Python's `ast` module is underused and genuinely powerful. Walk a Python file's AST, identify all function definitions (`ast.FunctionDef`), and for each function, record every other function it calls. The output is a directed graph: node = function, edge = "A calls B."

Implement your graph as an adjacency list: `dict[str, set[str]]` where keys are `module.function_name`. Write a `CodebaseParser` that walks a directory, parses each `.py` file, and builds the full graph.

Interesting queries to implement once the graph is built: Which functions are never called (dead code candidates)? Which functions are called by the most other functions (high coupling)? What's the maximum call depth from an entry point (longest call chain)?

**DSA focus:** This week is all about graphs. Implement **DFS** for cycle detection (mutual recursion = test complexity risk), **BFS** for finding all functions reachable from a given entry point, and **topological sort** for identifying a safe test execution order.

---

### Week 11 — Graph Algorithms and Coverage Gap Detection

**What to build:** A test coverage analyzer built on the call graph, not on line counts.

Standard coverage tools (pytest-cov) tell you which lines ran. Your tool tells you which *call paths* are never exercised by tests. The idea: parse the test files to identify which functions each test calls directly. Then use BFS on the call graph to find all transitively called functions. Any function that appears in no test's reachable set is a coverage gap.

Implement a `CoverageAnalyzer` that outputs: uncovered functions, functions only covered by one test (fragile coverage), and functions that are deeply nested in call chains (high blast radius — if they break, many tests fail).

**Design pattern:** **Observer pattern** for the analysis pipeline. An `AnalysisEvent` enum (`FILE_PARSED`, `FUNCTION_FOUND`, `EDGE_ADDED`, `ANALYSIS_COMPLETE`). Concrete observers: `ProgressLogger`, `CoverageTracker`, `ComplexityScorer`. The parser emits events; observers react. This makes it trivial to add new analysis types without modifying the core parser.

---

### Week 12 — LLM Integration and RAG Setup

**What to build:** A RAG pipeline that gives the LLM context about the codebase before generating test suggestions.

The naive approach — dump an entire file into an LLM prompt — fails for large codebases. Instead, build a RAG pipeline: chunk the codebase by function (each function + its docstring + its callers/callees = one chunk), embed each chunk, store embeddings in **ChromaDB** (local, no setup required), then at query time retrieve the k most relevant chunks and inject them into the prompt.

When asked "write tests for `process_payment`", your system: embeds the query, retrieves the function's code + its callees + any existing tests for similar functions, injects all of that as context, then prompts the LLM to generate targeted tests.

**Design pattern:** **Pipeline pattern** for the RAG flow: `chunk → embed → store` (indexing pipeline) and `query → retrieve → augment → generate` (query pipeline). Each stage is a composable step.

---

### Week 13 — Plugin System and Multiple Analyzers

**What to build:** A plugin architecture so the tool is extensible, plus two more analyzer types.

Implement a **Factory + Registry** plugin system. An `AnalyzerPlugin` abstract base class with `name: str`, `description: str`, `analyze(codebase: Codebase) -> AnalysisResult`. A `PluginRegistry` that discovers plugins via Python entry points (or simply by scanning a `plugins/` directory for classes that subclass `AnalyzerPlugin`). The CLI command `analyze --plugin coverage,complexity,llm` runs only the specified analyzers.

Implement two new plugins: `ComplexityAnalyzer` (cyclomatic complexity — count branches in each function's AST; functions above threshold are flagged as hard to test) and `DocstringAnalyzer` (functions missing docstrings, LLM generates suggested docstrings).

Add a `--report` flag that outputs a structured JSON report: each finding has a `severity` (LOW/MEDIUM/HIGH), `location` (file + line), `message`, and optional `suggestion`.

---

### Week 14 — Tests, Polish, README, and Ship

**What to build:** A properly tested, documented, installable CLI tool.

Make it installable: `pip install -e .` should expose a `codeanalyze` CLI command (use Click). Test it against three different open-source Python repos (grab something from GitHub) and include screenshots of the output in your README.

Write comprehensive tests — unit tests for each analyzer, integration tests against a small fixture codebase you include in the repo. Your test coverage for this project should be exemplary; it's an analyzer for tests, after all. The irony of having poor tests on a test analyzer tool would not be lost on a technical interviewer.

Your README should answer: What problem does this solve? How is it different from flake8/pylint/pytest-cov? How do I install and run it? What does the output look like? Include a terminal recording (use `asciinema`) showing it analyzing a real codebase.

Publish to PyPI (optional but impressive). Create a GitHub Actions workflow: lint, test, and optionally publish on tag push.

---

## Tips for All Three Projects

**Git discipline matters.** Commit frequently with meaningful messages. Use feature branches. Squash before merging. Reviewers and interviewers look at git history.

**Write about what you build.** Even short LinkedIn posts ("This week I implemented a decision tree from scratch — here's what I learned about information gain") build your presence and give you talking points. Two or three posts per project is plenty.

**Prioritize finishing over perfecting.** A shipped v1 with rough edges beats a perfect v0 that never launches. Ship each project on schedule even if you'd like to add more features.

**Tell the story.** For each project, write one paragraph in your `README.md` titled "Why I built this" — explain the motivation, what you learned, and what you'd do differently. This is what makes your portfolio human.
