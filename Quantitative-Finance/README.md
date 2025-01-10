## Overview
```
Note: These whole projects/tests are done by self, without any supervision. 
There is a limited access to the data, so strategies, may perform better or worse in live markets. 
Trading entails 100% risk to the capital, so consider personal limitations while taking action in live markets.
```
The goal is to test various strategies on widely available historical stock data. Each strategy can be further optimized, with personal bias and changing the input variables.

## Projects:
### Project 1: 3-day Price Change Strategy
----
- Description:
    - Enter long trade after 3 consecutive ~1%+ down days. Exit with +/- 1% target/SL.
    - Enter short trade after 3 consecutive ~1%+ up days. Exit with +/- 1% target/SL.
- Backtesting:
    - This strategy is back tested on 1-day historical interval data over multiple scrips/indexes.
- Limitations:
    - Limited access to data, further improvements can be made by adjusting the exit strategy to work on 15-min / 1-hr interval data.

- **Features:** 
    - For the whole study, primary focus is to build functions which can be used for similar other studies or compare results by changing parameters.
    - Please go through the `1_3-day_Change.ipynb` Jupyter Notebook, as the logic used while implementing each task is explained along with individual task solutions

### Project 2: Statistical Arbitrage
----
- Description:
    - Spread is calculated between two scrips under consideration.
    - Mean and standard deviation on spread is the next step which defines entry and exit.
    - Long entry is taken at Spread mean - deviation and short entry is taken at Spread mean + deviation.
    - Exit is at the crossover with Spread mean value.
- Backtesting:
    - This strategy is back tested on 1-day historical interval data.
- Limitations:
    - Limited access to data, further improvements can be made by adjusting the exit strategy to work on 15-min / 1-hr interval data.
- **Features:** 
    - For the whole study, primary focus is to build functions which can be used for similar other studies or compare results by changing parameters.
    - Please go through the `2_Statistical_Arbitrage.ipynb` Jupyter Notebook, as the logic used while implementing each task is explained along with individual task solutions.
- Evaluation Metrics: Cummulative Returns, Sharpe ratio, Maximum Drawdown

## To Do:

There are a few (planned) strategies to work on. But due to lack of access to data required to test the strategies, there is limited work done on the same.

Please find detailed descriptions in `Strategy_to_test.md`.

Happy to collaborate on the task, please email at: nikhil.wm27@gmail.com

## Usage

The best way to see the functions in action is to clone the repository.

Be mindful, you might need to install modules as per your environment. Best way to do this would be to run this in your terminal:
```Terminal
pip install <module>
```

## Contact
Thank you for dropping by, for any queries please feel free to contact on LinkedIn or by email

[LinkedIn](https://www.linkedin.com/in/nikhil-arora-6837501a4/) | [Email](nikhil.wm27@gmail.com)

Please have a look at my other Projects:

[GitHub](https://github.com/nikhil-hs27/Projects)