# Singapore HDB Resale Flat Analysis

By: Traci Lim

------

We will be looking at HDB resale flat transactions for the past 28 years, 1990 to 2018. The main motivation of this analysis is to get a hold on what the HDB resale market is like. In economics, we are interested in something called the equilibrium, where supply meets demand. This means that there's no surplus and no shortage of goods. If we think of housing as a good, well, it is a complex market because it is a good that cannot be built quickly, so perfect matching of supply with demand is not feasible. In this analysis, we will be looking at:

1. Some charts built from Tableau and Python
2. HDB resale flat suggestion tool
3. Predicting housing prices ([link](2_regression.ipynb))
4. Difference-in-differences model to estimate the impact of the opening of downtown MRT line have on housing prices

Visit the Tableau Dashboard [here](https://public.tableau.com/shared/DCJRRJKC9?:display_count=yes). 

------

![pricetrend](pictures\pricetrend.PNG)



The above shows a highlight table on the top left panel, which shows the yearly median prices of HDB resale prices sorted by towns. Just by plainly scanning the table, we can observe that estates like Bishan and Bukit Timah have consistently high prices. This is expected because they are mature estates with good proximity to the central area. The odd thing is the median prices of flats in central area from the year 2014 to 2015. There was a 2x jump in prices, while other estates did not display any behavior like that. Why is this the case?

A little bit of investigation leads to conclude that Pinnacle@Duxton HDB (super tall hdb near outram park MRT) is the culprit. The launch of Pinnacle is around 2009 and when the Minimum Occupancy Period (MOP) ended on Dec 2014, Pinnacle flat owners started to think that it is a good idea to sell them. Since Pinnacle itself is like the rolls royce of HDBs, the demand naturally picked up, that's why we see a spike in prices. You cant really argue with the tallest HDB in Singapore and its location. 

Next, we’re looking at the general median price trend of HDB resale flats in the bottom left panel. And notice that the price trend that we are observing here is simply the median price, which is not adjusted by inflation. It will be more accurate to look at the resale price index (RPI) in HDB’s website for a more reliable trend. But the RPI trend looks similar in pattern with our graphs here, the HDB resale price index increased by nearly 50% from early 2009 to 2013, before flatting out and then falling by 10% from 2013 to March-2017. 

If we look at the yearly trend on median resale prices across towns, we can observe a very distinctive slump in prices from 2013 to 2014,  where median prices suffers from some form of shock.  From 2009 to 2013, the general trend of median prices was growing rapidly, and this is likely due to the low interest rates, which translate to low Singapore interbank offered rate (SIBOR). For those who have taken out floating rate home loans, or for those coming out of fixed rate home loans, your interest rates typically follows the SIBOR or Swap Offer Rate (SOR). So in 2009, the SIBOR is at historic lows, at around 0.44. So this lasts all the way until 2015, when it started to rise with no signs of stopping. Even in 2018, the interest rates are set to increase. This is bad news for home owners.

As you may know, the Singapore government has always played a pivotal role in developing and managing the residential land and property market, ever since the 1960s. The government is primarily interested in making housing affordable for people. In Sep 2009, up to Dec 2013, when prices were growing rapidly, of course the government is worried. So looking at the number of transactions line chart, we can see drop in the number of flats purchase in the year 2010 to 2011, as well as 2012 to 2013. This is probably due to the cooling measures implemented by the government in that timeframe. In fact, they introduced a total of 10 cooling measures, which are policies that are aimed at relaxing the growing prices of the property market, so as to maintain affordability.

![transactions1](pictures\transactions1.PNG)

The government introduced a Seller’s stamp duty (SSD) for properties sold within 1 year of purchase, which was changed to 3 then 4 years years of purchase in 2011, that is payable by the home seller should he/she wants to sell their property. They lowered loan-to-value ratio (LTV) limits to In Dec 2011, which basically says you cannot borrow as much, so your downpayment just got bigger. They also introduced an Additional Buyer’s Stamp Duty (ABSD) that basically says now if you want to buy a 2nd home or 3rd home or if you’re a foreigner or PR, you pay more taxes. With the Mortgage service ratio, only 30% of a borrower’s income can be used to service the mortgage on HDBs or ECs. Some other stuff are Total Debt Servicing Ratio (TSDR), which limits the amount borrowers can spend on debt repayments to 60 percent of their gross monthly income, basically to ensure that  people borrow, and banks lend, responsibly. These debt repayments include car loans, credit card debt or personal loans. So, this is a rundown on the cooling measures that government have implemented. More recently, in 2018, ABSD went up again, and LTV limits went down. In other words, home buyers have to pay more taxes, and cannot borrow as much, so downpayment once again got bigger. Certainly, it looks like the cooling measures have worked, the prices finally came down after the government finished their ‘rapid firing’ of cooling measures, because they were implemented often within 6 months of one another. However, we must understand that those cooling measures mainly targeted at the private housing market. We know HDB concessionary loans are not affected by LTV limits and ABSD, but these cooling measures still managed to cause a two shocks in the number of transactions of HDB resale flats in year 2011 and 2013. This is most probably due to spillover effects of the impact on private housing market. As it becomes more costly to purchase new private properties, people who originally wanted to buy them would refrain from buying, causing the prices to fall in favor for first-time home buyers who wanted to buy a HDB, because they gained a second option---less expensive private condos. 

Whether or not the cooling measures made any substantial impacts on the housing market can be confirmed in two ways: informally or formally. Informally, one can just look at what happened to the price trend and draw some intuitions. The thing about informal methods of assessing the impacts of policies is that this is what we call eyeball econometrics. What we did up till now, are all considered eyeball econometrics. Eyeball econometrics cannot attribute causality to any of these policies. Because the housing market is a very complex market, there are a variety of macroeconomic factors that may affect the prices, that cannot be understood from just HDB resale transactions. Usually there will be researchers and ecometricians working on formal statistical methods to confirm the true impacts of these measures. But I doubt that that is going to be an easy task, because the cooling measures are implemented so closely within one another, it is near impossible for these econometricians to control for biases. 

------

### HDB Resale Flat Suggestion Tool

With the cleaned and preprocessed data, a HDB resale flat suggestion tool will possibly be useful to some extent. We’ll be looking at a dashboard that suggests flat types and towns suited for a user’s input budget. You can optimize your preferences according to how close is the flat to the nearest MRT station or school. The dashboard will run through the data and output a highlight table according to your preferences, which shows the median prices of resale units for 2017 and 2018 transactions. I managed to input a distance option into the tool by calculating the distance to the nearest MRT station and school for all HDBs in the data. All preprocessing code can be view in.

![tool](pictures\tool.PNG)

------

### Policy Analysis using Difference-in-Differences model

Let’s think about this. How should we conduct a formal statistical test, as an econometrician, to investigate whether the opening of DTL affects the prices of houses that are near DTL stations? We don’t want to just look at houses in Bukit Panjang estate, we want to look at all flats near the DTL stations. The formal statistical test is difference in differences model, which is one of the methods used by econometricians to estimate the causal effects of a policy. 

In other words, this Diff-in-diff model is going to tell us, how much of an impact did the opening of DTL make on HDB resale prices. To set up our model, we need to select 2 groups, a control and a treatment group. The treatment group is group of flats that we think will benefit from the opening of DTL, the control group is a group of flats that we think will be indifferent to the opening of DTL. More specifically, we put all flats within 1km radius of any DTL station into the treatment group, and all flats outside 1km radius of any DTL station into the control group. 

![did1](pictures\did1.png)

We are using data on the two groups, before and after the treatment occurs. The diff-in-diff estimate calculates the difference in outcomes before and after the treatment. Then, the difference between those two differences is the estimated effect of the treatment. So, did the opening of DTL affect average price per sqm of HDB resale flats? The reason we’re looking at the average price per sqm instead of resale prices, is because average price per sqm suffers from a lower variance than resale prices. So the spread of values is bigger in resale prices than in price per sqm, especially since we are talking about the spread of prices in all flat types. So I think using average price per sqm here is more appropriate. The average price per sqm is the average price per sqm of all flats in that month, be it a 5-room, or 2-room, or 10 storey, or Yishun, or Central area. So the number of flats in the control group is around 100k, and number of flats in treatment group is around 500k. 

![did2](pictures\did2.PNG)

We plot log-average-price-per-sqm with respect to the sale time, which is indicated by the number of months starting from the first sample date of Jan 2010. The curves show the price gradient for houses in the control and treatment groups, fitted by a higher-order polynomial regression. To put it in layman terms, we are looking a plot of average price per sqm points, and then a best fit line is drawn to match these points, similar to linear regression, except in case, the line is not straight, it is curvy. 

It is clear that from month 10 to 48, the average price per sqm of resale flats in the treatment group attained relatively higher prices than of the control group. Since DTL was not opened in that period of time, we can safely deduce that this is a result of anticipative effects of the public housing market prior to the opening of DTL. So, the announcement of the plan to open DTL itself could have made an impact on average price per sqm of houses. The opening of DTL is separated into 3 stages, every stage opens a select bunch of stations. From month 49 onwards, after the opening of DTL line in stage 1, the two price gradients diverged during Dec 2013. when some stations in DTL became operational ready. The gap in prices continued to grow until stage 2, and then it starts to stabilize and move in tandem with the control group. For all diff-in-diff models, our selection of control and treatment groups have to satisfy something called the parallel trends assumption. The parallel trends assumption means that the gap between the treated and control group would've remained the same if there is no treatment given to the treatment group. This plot can affirmed this assumption, as you can see, before stage 1, the price gradients of 2 groups follows a similar trend. This protects us from a certain amount of selection bias, though we cannot say for sure that we have eliminated all forms of selection bias. 

Now that we’ve got a good idea of the price trends of control and treatment groups, let's do a formal statistical test. So the diff-in-dIff estimator is as follows, this a simple model that is fitted with a regression on 3 variables, Dummy variables ```treated``` and ```time```, and the interaction variable ```did```. It is pretty simple to interpret this. Look at this values in the outline box, this means that average price per sqm of flats grew by 3.9% for flats within 1km radius of a DTL station, after the stage 1. But notice that, the average price per sqm of flats was already experiencing a 2.7% growth after stage 1, regardless of whether it is close to a DTL station or not. **We can therefore conclude that opening of DTL did contribute to the growth of prices of flats within 1km to a DTL station, 1.2% to be exact.** 

So, as for every statistical model, there are weaknesses and strengths. For this model, it does not control for other housing attributes like storey range, town, flat type, or location attributes, like distance to nearest MRT, distance to city. We can definitely build a better model that controls for such attributes. So instead of regressing on just 3 variables, we can add additional features.

------

### Conclusion

After the financial crisis in 2008, which was caused by over-extending credits to housing loans leading to bursting of housing bubbles, many countries have learnt to avoid repeating the same mistake. To avoid this, Singapore will always try to achieve long term stabilization in property markets via cooling measures. We can speculate that Singapore's housing market will be regularly treated with cooling measures to keep housing affordable. While first-time home buyers do not need to worry about these, this could spell bad news of 2nd/3rd home buyers, or anybody not taking a HDB loan.