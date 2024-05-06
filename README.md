# Final Project Team 5 - Neural Networks Spring 2024

## Members
- Ziyi Guo
- Jeremy Whipple
- Jessica Cisneros
- Adam Shaker

## Resources
- [Google Drive](https://docs.google.com/document/d/192plAoD7LcFVFtnl3MZ80Ufv4Z3IQYNonejYrzYmdpo/edit?usp=sharing)

### Goals
- [ :white_check_mark: ] Complete the `Single Perceptron Training`
- [ :white_check_mark: ] Complete the `Module-7 Inspired Neural Network Training`
- [ :white_check_mark: ] Complete Optimization of threshold
- [ :white_check_mark: ] Complete Report

### BreakDown/Goals
- Train the network using the **odd-numbered** data points and the online training technique with 30 cycles
- Optimize the trained networks by determining the best threshold for mapping the output to 0 or 1, and evaluate the Receiver Operating Characteristics (ROCs).
- Test the networks using the **even-numbered** data points to determine the actual ROCS.
- Write a report documenting the network designs, computational performance, analysis, and suggestions for improvement, following the provided guidelines and grading criteria.

The report contains the following:
- **Executive Summary** - The first page. Summarizes the problem, our solution and our results - **Finish after completing analysis**
- **Introduction** - Introduce and describe the problem - and what we would hope our solution would solve, Detail the two networks we have, Describe how they solve the problem. - **Can be finished now**
- **Model Performance** - Details the performance of the models we trained and compares them. (Computational Performance? - `Note` not sure what this is supposed to be) - **Complete after choosing Model Threshold**
- **Performance Analysis** - Analyze the performance of the models, include details about the metrics we chose and how we determined our Threshold. - **Complete after analyzing performance**
- **Conclusion** - Summarize the problem, networks, and performance and discuss possible improvements (can just be conjecture). - **Complete after analysis**
- **Appendix** - Contains images and code as well as other references we may have made use of in our efforts. - **Add to as we work on the report**

**NOTE** - The difference between the two 'performance' sections isn't terribly clear, so we don't need to separate them so much.  Instead we can have a section discussing the performance and comparison betwwen the two models, including the metrics and 'analysis' relating to those (We need to discuss all our observations). We can then also discuss our thoughts for the possible improvments in a different section.

#### Summary of the Project
- Advertising companies want to maximize revenue per dollar spent on targeted advertising campaigns.
- Two Important Features:
  - Size of Wallet (SOW): An estimate of a household's disposable income.
  - Local Affluence Code (LAC): A measure of neighborhood affluence based on housing prices and other factors.
  - These features map to the Targeted Advertising Code Assignment (TACA), which indicates whether a targeted advertising campaign is likely to have a positive (1) or negative (0) return.
