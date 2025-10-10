# Pipeline
## Old pipeline
    +-------------+         +-------------+         +-------------+        +-------------+       +-------------+
    |             |         |             |         |             |        |             |       |             |
    | raw letters | ------> |   segment   | ------> |   classify  | -----> |   detect    | ----> |  horizontal |
    |             |         |   letter    |         |   angle     |        |   receiver  |       |  receiver   |
    |             |         |             |         |             |        |             |       |             |
    +-------------+         +-------------+         +-------------+        +-------------+       +-------------+

## New pipeline
    +-------------+         +-------------+         +-------------+        +-------------+
    |             |         |             |         |             |        |             |
    | raw letters | ----->  |   detect    | ------> |   classify  | -----> |  horizontal |
    |             |         |   OBB       |         |   angle     |        |  receiver   |
    |             |         |   receiver  |         |             |        |             |
    |             |         |             |         |             |        |             |
    +-------------+         +-------------+         +-------------+        +-------------+


New pipeline is faster, only 4 steps instead of 5 steps which reduce to only 70% of the time taken by old pipeline.