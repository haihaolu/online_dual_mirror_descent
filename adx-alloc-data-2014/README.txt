Description of the Data Files for "Yield Optimization of Display Advertising with Ad Exchange"
Santiago Balseiro, 2014.

In order to preserve the anonymity of the publishers in these files, names have been removed and 
all the values have been multiplied by a random constant.

We selected representative publishers of different size: two small publishers with around 
10 contracts, two large with around 100 contracts, and three medium size publishers in between. The
publishers are mostly online gaming websites, and news websites. The data sets were collected over 
a period of one week during March of 2010, and the number of impressions in each data set 
ranges from 300 thousand to 7 million. Additionally, the fraction of impressions reserved for 
contracts varies across publishers, with two publisher highly constrained, one moderately 
constrained and the remaining lowly constrained. The following table summarizes the publishers 
information. Please read the paper for more details.

Pub # 	A	T	sum rho	Size	Cons.	Mean EQ	 Num. of Impr	s*(0)	R(0)
1	6	10	21%	Small	Low	582	 1,500,000 	68%	622
2	12	7	89%	Small	High	47	 2,100,000 	100%	448
3	17	13	43%	Medium	Medium	820	 320,000 	74%	1883
4	17	15	28%	Medium	Low	686	 930,000 	76%	1320
5	29	27	73%	Medium	High	1152	 1,800,000 	99%	1424
6	98	173	28%	Large	Low	542	 6,700,000 	75%	2076
7	101	406	16%	Large	Low	209	 7,000,000 	67%	1378


===================================================================================================
ADFILE (pubX-ads.txt):
=================================================================================================== 
File with ratio of capacities to number of impressions of the advertisers. One row per advertiser
with the following format:

advertiser: int rho: float

The field <advertiser> is the id of the advertiser and <rho> is the capacity ratio.

===================================================================================================
TYPEFILE (pubX-types.txt):
===================================================================================================
File with the distribution of the types. One per per type with the following format:

type: int prob: float advertisers: [int,...] mean: [float,...] cov: [float, ...] 

The field <type> is the id of the type and <prob> if the arrival probability of that type. The field
<advertisers> specifies the ids of the advertisers whose criteria matches the type. The field <mean>
specifies the mean vector of the log-normal distribution of the quality perceived by the 
qualifying advertiser (should have the same length that the <advertisers> sequence). The field <cov>
specifies the upper triangular part of the covariance matrix by column (column major order) of the 
log-normal distribution. 

===================================================================================================
ADXFILE (pubX-adx.txt):
===================================================================================================
File with the empirical complement of the quantile of the highest bid and expected revenue function.
The file has a header:

accept.prob price revenue

The file should have three columns of numbers separated by white spaces. <accept.prob> is the 
probability that the impression is accepted by the AdX given that the reserve is <price>, i.e, there
is a bid larger than the price. <revenue> is the expected revenue under that reserve <price> or 
<accept.prob>.