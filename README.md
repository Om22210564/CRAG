# Corrective Retrieval-Augmented Generation (CRAG)

```
                            User Query
                                 │
                                 ▼
                            Retrieval
                                 │
                                 ▼
                       Retrieval Evaluator
                                 │
              ┌──────────────────┼────────────────────┐
              │                  │                    │
              ▼                  ▼                    ▼
           CORRECT            AMBIGUOUS            INCORRECT
              │                  │                    │
              ▼                  ▼                    ▼
      Knowledge Refinement     Knowledge           Discard local
              │               Refinement               │
              │                  │                     │
              │                  ▼                     ▼
              │           Web Search(rewrite q)   Web Search(rewrite q)
              │                  │                     │
              │                  ▼                     │
              │             Web Results                │
              │                  │                     │
              │                  ▼                     │
              │           Knowledge Refinement         │
              │                  │                     │
              └──────────────────┼─────────────────────┘
                                 ▼
                            Generation
                                 │
                                 ▼
                               Answer
```
