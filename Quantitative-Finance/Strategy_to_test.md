# Strategy to test 1:
- **Title:** Targeted Short Hedge (TSH)
- **Description (Initial Test):**
    - Entry:
        - Instrument: Options only
        - Spot Strike price: nearest to cmp
        - Expiry: Monthly Expiry or atleast third week expiry from current date
        - Entry conditions: A reasonable logic for a up-move or down-move in the scrip/index
        - Entry:
            - Expected Up-move:
                - Position: Short PE 2.5% ITM from spot
                - Hedge:    Long PE 2.5% OTM from spot
            - Expected Down-move:
                - Position: Short CE 2.5% ITM from spot
                - Hedge:    Long CE 2.5% OTM from spot
            - Qty: Same for target and hedge
    - Holding Period: Till 1-week from expiry
    - Exit:
        - Target is to make money on short trade only. Goal is to keep time-value on our side.
        - Exit:
            - If market moves towrads the expected direction:
                - Exit with 1.5x-3x profit on position
                - Take the loss on hedge
            - If market moves against the expected direction:
                - Hold till Strike price of Hedge is breached.
                - Exit both position and hedge with 0-10% loss on margin
            - Compulsory exit: 5 trading days from expiry. Take the whole Loss.
        - Motivation: Survive to fight again another day. (Protect Capital)
``` 
Backtest Limitation:
    - Don't have access to the kind of data required to perform backtesting.
```
- Furthering testing/improvements:
    - Setting entry/logic:
        - Set a user-input Support/resistence levels parameter to decide entry/exit.
        - Work-around possible VIX values to set entry.
        - Work-around possible greeks to set entry.
    - Test with different strike prices for position and hedges.