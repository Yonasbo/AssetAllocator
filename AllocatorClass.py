import pandas as pd
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import EfficientFrontier , EfficientCVaR , EfficientSemivariance, risk_models, expected_returns
import quantstats as qs
from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from time import localtime, strftime
import os
qs.extend_pandas()


def DataLoader(date:str='2023/08/24',handlingNa:int=-1)-> tuple:
    """
    Data can modified such they end at a specific timestamps, rows with less entries than given param
    are droped, rest missing entries are forward filled, if applicable. Then loaded into specific 
    DataFrames which are the output. 
    
    Input:
        :param date: (optional) date to when the data should end, if not given whole data will be returned
        :type date: str
        :param handlingNa: (optional) require that many non-NA values in a row, if not given no rows will be dropped
        :type handlingNa: int
    
    Return:
        :param df_protection: historical data of the protection asset
        :type df_protection: pd.DataFrame
        :param df_benchmark: historical data of the benchmark index
        :type df_benchmark: pd.DataFrame
        :param df_invest: historical data of the investment assets
        :type df_invest: pd.DataFrame
        :param df_risk: historical data of the risk indices
        :type df_risk: pd.DataFrame
    
    """
    df_data = pd.read_csv('Data_DeutscheBank.csv', parse_dates=True, index_col="Date")
    df_data = df_data[df_data.index <pd.to_datetime(date) ] 
    if handlingNa==-1:
        df_data.fillna(method='ffill',inplace=True)
    else:
        df_data.dropna(thresh=handlingNa,inplace=True)
        df_data.fillna(method='ffill',inplace=True)

    Signs = {'benchmark': ['SPX'],
         'Risk' : ['VIX', 'DBMARMSS'],
         'ProtectionAsset' : ['LU0099730524'],
         'InvestmentAssets' : [
             'LU2034326236',
             'LU0292107645',
             'IE00BQXKVQ19',
             'LU0397221945',
             'DE0008490962',
             'LU1978535810',
             'LU0411078552',
             'LU0908508814',
             'LU0641007009',
             'LU0429459356',
             'LU0429458895'
         ]
         }
    df_benchmark = df_data[Signs['benchmark']]
    df_risk = df_data[Signs['Risk']]
    df_invest = df_data[Signs['InvestmentAssets']]
    df_protection = df_data[Signs['ProtectionAsset']]
    
    return  df_protection, df_benchmark, df_invest, df_risk



class BaseAllocater(ABC):
    """
    BaseAllocator is a abstract parent class to allocation and optimization classes. Containing multiple public 
    and private methods giving a framework to track changes in the allocated portfolio. Child classes 
    need to add a calculate_next_allocation method.


    Instance variables:
        -''dfProtection'' - DataFrame
        -''dfInvest'' - DataFrame
        -''dfBenchmark'' - DataFrame
        -''dfRisk'' - DataFrame
        -''investmentAmount'' - int
        -'startDate'' - str
        -''runtime'' - int
        -''riskFreeRate'' - float


    Public methods:
        -''get_timed_DataFrames(date)'' returns copies of the 4 DataFrame up to a specific date 
        -''get_new_portfolio_value(date)'' calculates new portfolio value for a given date
    """
    def __init__(
            self,
            dfProtection:pd.DataFrame,
            dfInvest:pd.DataFrame,
            dfBenchmark:pd.DataFrame,
            dfRisk:pd.DataFrame,
            investmentAmount:int = 100000000,
            startDate:str = '2007-01-01',
            runtime:int = -1,
            riskFreeRate:float = 0.02,
            fee:float = 0.001,
            deductFee:bool = True
    ) -> None:
        """
        :param dfProtection: Data for the protection asset.
        :type dfProtection: pd.DataFrame
        :param dfInvest: Data for the investment assets.
        :type dfInvest: pd.DataFrame
        :param dfBenchmark: Data for the benchmark index.
        :type dfBenchmark: pd.DataFrame
        :param dfRisk: Data for the risk indices.
        :type dfRisk: pd.DataFrame
        :param investmentAmount: the amount of the starting investment.
        :type investmentAmount: int
        :param startDate: first day of backtesting should be in the format %Y-%m-%d.
        :type startDate: str
        :param runtime: how many days we are backtesting, if the value is -1 the backtest runs in full
        :type runtime: int
        :param riskFreeRate: risk-free rate of borrowing/lending, defaults to 0.02.
        :type riskFreeRate: float
        :param fee: product fee charged by the bank
        :type fee: float
        "param deductFee: should the fee be substracted monthly from the portfolio
        :type deductFee: bool
        """


        self._dfProtection = deepcopy(dfProtection)
        self._dfInvest = deepcopy(dfInvest)
        self._dfRisk = deepcopy(dfRisk)
        self._dfBenchmark = deepcopy(dfBenchmark)
        self.portfolioValue = investmentAmount
        self.startDate = pd.to_datetime(startDate)
        self._runtime = runtime
        timeframe = dfProtection[dfProtection.index >= self.startDate].index if dfProtection is not None else None
        self.timeframe = timeframe if runtime<0 else timeframe[:runtime]  if dfProtection is not None else None
        self.highWatermark = investmentAmount
        self.maxDrawdown = 0
        self.riskFreeRate = riskFreeRate
        columns = ['Portfolio Value','Investment Assets percent']+list(self._dfProtection.columns+[' Units'])+list(self._dfInvest.columns+[' Units'])
        lastPrice = get_latest_prices(self._dfProtection[self._dfProtection.index <=self.timeframe[0] ])

        self._lastAllocation , self._lastLeftover = DiscreteAllocation({self._dfProtection.columns[0]: 1.0},
                                                                     lastPrice,total_portfolio_value=self.portfolioValue).greedy_portfolio()
        self._month = self.startDate.month
        self._year = self.startDate.year 
        self._fee = fee
        self._monthlyFees = {}
        self._deductFee = deductFee

        self.portfolioMetrics = pd.DataFrame(columns=columns)
        self._daily_portfolio_changes(self.timeframe[0])
        

    def set_dfProtection(self, dfProtection:pd.DataFrame) -> None:
        self._dfProtection = deepcopy(dfProtection)
        self._set_timeframe()
        
    def set_dfInvest(self, dfInvest:pd.DataFrame) -> None:
        self._dfInvest = deepcopy(dfInvest)

    def set_dfRisk(self, dfRisk:pd.DataFrame) -> None:
        self._dfRisk = deepcopy(dfRisk)

    def set_dfBenchmark(self, dfBenchmark:pd.DataFrame) -> None:
        self._dfBenchmark = deepcopy(dfBenchmark)

    def get_dfProtection(self) -> pd.DataFrame:
        return deepcopy(self._dfProtection)
    
    def get_dfInvest(self) -> pd.DataFrame:
        return deepcopy(self._dfInvest)
    
    def get_dfRisk(self) -> pd.DataFrame:
        return deepcopy(self._dfRisk)
    
    def get_dfBenchmark(self) -> pd.DataFrame:
        return deepcopy(self._dfBenchmark)


    def _set_timeframe(self) -> None:
        '''
        Helper method to set the timeframe
        '''
        timeframe = self._dfProtection[self._dfProtection.index >= self.startDate].index if self._dfProtection is not None else None
        self.timeframe = timeframe if self._runtime<0 else timeframe[:self._runtime] if self._dfProtection is not None else None
    
    def get_timed_DataFrames(self, date:pd.Timestamp) -> tuple:
        """
        Get the 4 dataframes up to specific date

        :return: Tuple of Dataframes where all entries are from dates <= date
        :type: pd.DataFrame
        """

        if not isinstance(date,pd.Timestamp):
            date = pd.to_datetime(date)
        dfProtection = deepcopy(self._dfProtection)
        dfInvest = deepcopy(self._dfInvest)
        dfBenchmark = deepcopy(self._dfBenchmark)
        dfRisk = deepcopy(self._dfRisk)

        dfProtection = dfProtection[dfProtection.index <= date]
        dfInvest = dfInvest[dfInvest.index <= date]
        dfBenchmark = dfBenchmark[dfBenchmark.index <= date]
        dfRisk = dfRisk[dfRisk.index <= date]
    
        for ticker in dfInvest.columns:
            if dfInvest.tail(10).isna()[ticker].any():
                dfInvest.drop(ticker,axis='columns',inplace=True)

        return dfProtection, dfInvest, dfBenchmark, dfRisk

    def _update_high_watermark(self) -> None:
        """
        Helper method to update the Highwatermark
        """
        if self.highWatermark<self.portfolioValue:
            self.highWatermark = self.portfolioValue

    def _update_max_drawdown(self) -> None:
        """
        Helper method to update the maximum Drawdown
        """
        if self.maxDrawdown < (self.highWatermark-self.portfolioValue)/self.highWatermark*100:
            self.maxDrawdown = (self.highWatermark-self.portfolioValue)/self.highWatermark*100
    
    def get_new_portfolio_value(self,date:pd.Timestamp) -> None:
        """
        Calculates the new portfolio value for a date.
        :param date: date 
        :type date: pd.Timestamp
        :raises ValueError: if ''date'' is not a pd.Timestamp
        """
        if not isinstance(date, pd.Timestamp):
            raise ValueError("date should be a pd.Timestamp")

        lastPrice_prot = get_latest_prices(self._dfProtection[self._dfProtection.index <=date ])
        latestPrices_invest = get_latest_prices(self._dfInvest[self._dfInvest.index <=date ])
        latestPrices = pd.concat([latestPrices_invest,lastPrice_prot],axis=0)
        self.portfolioValue = sum([self._lastAllocation[index]*latestPrices[index] for index in self._lastAllocation])+self._lastLeftover

        self._update_high_watermark()
        self._update_max_drawdown()
    
    
    def _monthly_fee_deduction(self, date:pd.Timestamp) -> None:
        """
        Helper method to calculate the monthly fee and update the portfolio accordingly
        :param date:
        :type date: pd.Timestamp
        :raises ValueError: if ''date'' is not a pd.Timestamp
        """
        if not isinstance(date, pd.Timestamp):
            raise ValueError("date should be a pd.Timestamp")
        pfMetrCopy = deepcopy(self.portfolioMetrics)
        month_slicer = (pfMetrCopy.index.month ==self._month) & (pfMetrCopy.index.year == self._year)
        mFee = (pfMetrCopy[month_slicer]['Portfolio Value']*pfMetrCopy[month_slicer]['Investment Assets percent']).mean()*self._fee/12
        if self._deductFee:
            self.portfolioValue-=mFee
        self._monthlyFees[date] = mFee
        self._month = date.month
        self._year = date.year

    def _daily_portfolio_changes(self,date:pd.Timestamp) -> None:
        """
        Helper method to calculate daily changes in the portfolio. Tracks the Prtfolio value,
        how much is in investment assets and how many units of each assets is in the portfolio.
        :param date:
        :type date: pd.Timestamp
        :raises ValueError: if ''date'' is not a pd.Timestamp
        """
        if not isinstance(date, pd.Timestamp):
            raise ValueError("date should be a pd.Timestamp")

        columns = self.portfolioMetrics.columns
        dailyPortfolioMetrics = pd.DataFrame([[0]*len(columns)],index=[date],columns=columns)
        for ticker in self._lastAllocation:
            for column in columns:
                if ticker in column:
                    dailyPortfolioMetrics.at[date,column] = self._lastAllocation[ticker]

        lastPrice_prot = get_latest_prices(self._dfProtection[self._dfProtection.index <=date ])
        latestPrices_invest = get_latest_prices(self._dfInvest[self._dfInvest.index <=date ])
        latestPrices = pd.concat([latestPrices_invest,lastPrice_prot],axis=0)
        portfolioValue = self.portfolioValue

        dailyPortfolioMetrics.at[date,'Portfolio Value'] = portfolioValue
        allo = deepcopy(self._lastAllocation)
        prot_name = self._dfProtection.columns[0]
        if prot_name in allo:
                del allo[prot_name]
        percentInvest = sum([allo[index]*latestPrices[index] for index in allo])/portfolioValue

        dailyPortfolioMetrics.at[date,'Investment Assets percent']  = percentInvest

        self.portfolioValue = portfolioValue
        self.portfolioMetrics = pd.concat([self.portfolioMetrics,dailyPortfolioMetrics],axis=0)
    
    @abstractmethod
    def calculate_next_allocation(self, date:pd.Timestamp) -> None:
        pass

class EfficientFrontierAllocator(BaseAllocater):
    """
    EfficientFrontierAllocator is an object (inheriting from BaseAllocator and using funtions & classes 
    from PyPortfolioOpt) for allocation and optimization of a portfolio. Containing multiple public 
    and private methods giving a framework to track changes in the allocated portfolio.


    Instance variables:
        -''dfProtection'' - DataFrame
        -''dfInvest'' - DataFrame
        -''dfBenchmark'' - DataFrame
        -''dfRisk'' - DataFrame
        -''investmentAmount'' - int
        -'startDate'' - str
        -''runtime'' - int
        -''riskFreeRate'' - float
        -''optimizer'' - str 
        -''riskModel'' - str 
        -''expectedReturnsModel'' - str
        -''split'' - float
        -''volatilityWindow'' - int
        -''volBound'' - int 

    Public methods:
        -''get_timed_DataFrames(date)'' returns copies of the 4 DataFrame up to a specific date 
        -''get_new_portfolio_value(date)'' calculates new portfolio value for a given date
        -''calculate_next_allocation(date)'' calculates the next allocation of assets for a date
    """
    def __init__(
            self, 
            dfProtection:pd.DataFrame, 
            dfInvest:pd.DataFrame, 
            dfBenchmark:pd.DataFrame,
            dfRisk:pd.DataFrame,
            investmentAmount:int = 100000000, 
            startDate:str = '2007-01-01', 
            runtime:int = -1,
            riskFreeRate:float = 0.02,
            fee:float = 0.001,
            deductFee:bool = True,
            optimizer:str = 'meanVar',
            riskModel:str = 'ledoit_wolf',
            expectedReturnsModel:str = 'capm',
            split:float = .5,
            volatilityWindow:int = 9,
            volBound:int = 23
    ) -> None:
        """
        :param dfProtection: Data for the protection asset.
        :type dfProtection: pd.DataFrame
        :param dfInvest: Data for the investment assets.
        :type dfInvest: pd.DataFrame
        :param dfBenchmark: Data for the benchmark index.
        :type dfBenchmark: pd.DataFrame
        :param dfRisk: Data for the risk indices.
        :type dfRisk: pd.DataFrame
        :param investmentAmount: the amount of the starting investment.
        :type investmentAmount: int
        :param startDate: first day of backtesting should be in the format %Y-%m-%d.
        :type startDate: str
        :param runtime: how many days we are backtesting, if the value is -1 the backtest runs in full
        :type runtime: int
        :param riskFreeRate: risk-free rate of borrowing/lending, defaults to 0.02.
        :type riskFreeRate: float
        :param fee: product fee charged by the bank
        :type fee: float
        "param deductFee: should the fee be substracted monthly from the portfolio
        :type deductFee: bool
        :param optimizer: optimization method to use from Mean Variance, Semivariance or CVar
        :type optimizer: str
        :param riskModel: risk model to use for covariance calculation
        :type riskModel: str
        :param expectedReturnsModel: method to calculate expected returns of the asset
        :type exprectedReturnsModel: str
        :param split: in case of high volatile market split between funds in protection asset and investment assets
        :type split: float
        :param volatilityWindow: how far the VIX is looking back in the past to calc a mean
        :type volatilityWindow: int
        :param volBound: upper bound for the VIX mean
        :type volBound: int
        :raises NotImplementedError: if 'optimizer' is not from 'meanVar','semiVar','cVar'
        :raises NotImplementedError: if 'riskModel' is not from 'sample_cov', 'semicovariance','exp_cov','ledoit_wolf'
        :raises NotImplementedError: if 'expectedReturnsModel' is not from 'mean','ema', 'capm'
        """

        super().__init__(
            dfProtection, 
            dfInvest, 
            dfBenchmark, 
            dfRisk, 
            investmentAmount, 
            startDate, 
            runtime,
            riskFreeRate,
            fee,
            deductFee
        )
        self.optimizer = optimizer if optimizer in ['meanVar','semiVar','cVar'] else 0
        if self.optimizer == 0: raise NotImplementedError("Optimizer {} not implemented".format(optimizer))
        self.riskModel = riskModel if riskModel in ['sample_cov', 'semicovariance','exp_cov','ledoit_wolf'] else 0
        if self.riskModel == 0: raise NotImplementedError("Risk Model {} not implemented".format(riskModel))
        self.expectedReturnsModel = expectedReturnsModel if expectedReturnsModel in ['mean','ema', 'capm' ] else 0
        if self.expectedReturnsModel == 0:raise NotImplementedError("Risk Model {} not implemented".format(expectedReturnsModel))
        self.split = split
        if self.split > 1 or self.split < 0: raise ValueError("Split of {} not in [0,1]".format(split))
        self.volatilityWindow = volatilityWindow
        self.volBound = volBound
        #count the consecutiv occurence of both risk indicators happening at the same time, 
        # gets reset everytime we use max_sharpe to optimize
        self._counter = 0 
        self._windowforcounter = 2

    def _daily_volatility_bound(self, volMean:float) -> float:
        """
        Helper method to calculate the volatility bound
        """
        volatility = 30 if self.highWatermark==self.portfolioValue else min(self.volBound,volMean)
        return volatility

    def _choose_next_optimization_strat(self,dfRisk:pd.DataFrame) -> tuple:
        """
        Helper method to calculate the next optimization strategy and if we have to 
        park some or all assets in the protection asset

        :param dfRisk: data of the VIX and DBMARMSS indices
        :type dfRisk: pd.DataFrame
        :return: Strategy and Binary indicator 
        :type: str and Bool
        """
        volMean = dfRisk.tail(self.volatilityWindow)['VIX'].mean()
        
        volatility = self._daily_volatility_bound(volMean)

        if dfRisk.iloc[-1]['VIX'] > volatility and dfRisk.iloc[-1]['DBMARMSS'] == 1:
            return 'min_vol' , True
        elif dfRisk.iloc[-1]['VIX'] > volatility or dfRisk.iloc[-1]['DBMARMSS'] == 1:
            return 'min_vol', False
            
        else:
            return 'max_sharpe', False

    def _calculate_expected_returns(self, dfInvest:pd.DataFrame, dfBenchmark:pd.DataFrame) -> pd.Series:
        """
        Helper method to calculate the expected return to a specific method, portfolio, and benchmark

        :return: expected Returns
        :type: pd.Series
        """
        
        if self.expectedReturnsModel == 'capm':
            expectedReturns = expected_returns.capm_return(dfInvest,dfBenchmark,risk_free_rate=self.riskFreeRate)
        elif self.expectedReturnsModel == 'ema':
            expectedReturns = expected_returns.ema_historical_return(dfInvest)
        else:
            expectedReturns = expected_returns.mean_historical_return(dfInvest)

        return expectedReturns

    def _calculate_weights(self, dfInvest:pd.DataFrame, dfBenchmark:pd.DataFrame,strat:str, solver='SCS', verbose=False) -> dict:
        """
        Helper method to calculate the weights of our allocation via the efficient frontier,
        ueses methods and classes from PyPortfolioOpt https://pyportfolioopt.readthedocs.io/en/latest/UserGuide.html
        
        :param dfInvest: investment Asset historical data
        :type dfInvest: pd.DataFrame
        :param dfBenchmark: Benchmark e.g. SPX, historical data
        :type dfBenchmark: pd.DataFrame
        :param strat: what we are optmizing for e.g. min volatilty or max sharpe
        :type strat: str

        :return: weights
        :type: dict

        """

        expectedReturns = self._calculate_expected_returns(dfInvest,dfBenchmark)
            
        historical_returns = expected_returns.returns_from_prices(dfInvest)
        CovMatrix = risk_models.risk_matrix(dfInvest, method = self.riskModel)
        if strat=='max_sharpe':
            EfficientFront = EfficientFrontier(expectedReturns, CovMatrix,solver=solver,verbose=verbose)
            self._counter =0
            raw_weights = EfficientFront.max_sharpe()
        else:
            if self.optimizer == 'meanVar':
                
                EfficientFront = EfficientFrontier(expectedReturns, CovMatrix,solver=solver,verbose=verbose)
                
                raw_weights = EfficientFront.min_volatility()
            elif self.optimizer =='cVar':
                
                EfficientFront = EfficientCVaR(expectedReturns,historical_returns,solver=solver,verbose =verbose)
                raw_weights = EfficientFront.min_cvar() 
            elif self.optimizer == 'semiVar':
                EfficientFront = EfficientSemivariance(expectedReturns,historical_returns,solver=solver,verbose=verbose)
                raw_weights = EfficientFront.min_semivariance()

        weights = EfficientFront.clean_weights()
        weights = { i: weights[i]/sum(weights.values()) for i in weights  }
        return weights

    def calculate_next_allocation(self, date:pd.Timestamp, solver='SCS', verbose=False) -> None:
        """
        Calculates the next allocation of assets for a specific date. 

        :param date: date of the new allocation
        :type date: pd.Timestamp
        :raises ValueError: if ''date'' is not a pd.Timestamp
        """
        if not isinstance(date, pd.Timestamp):
            raise ValueError("date should be a pd.Timestamp")
        
        if date.month!=self._month:
            self._monthly_fee_deduction(date)

        dfProtection, dfInvest, dfBenchmark, dfRisk = self.get_timed_DataFrames(date)
        strat, split_option = self._choose_next_optimization_strat(dfRisk)

        if split_option and self.split >0:
            self._counter+=1
            weights = self._calculate_weights(dfInvest,dfBenchmark,strat,solver,verbose)
            lastPrice = get_latest_prices(dfProtection)
            if self._counter>=self._windowforcounter:
                splitter = .995
            else:
                splitter = self.split
            allo , left = DiscreteAllocation({'LU0099730524': 1.0},lastPrice,total_portfolio_value=self.portfolioValue*splitter).greedy_portfolio()
            
            if splitter==1:
                allocation , leftover = allo, left
            else:
                latestPrices = get_latest_prices(dfInvest)
                investA = self.portfolioValue*(1-splitter) + left.round(2)
                discAllocation = DiscreteAllocation(weights,latestPrices, total_portfolio_value=investA)
                allocation, leftover = discAllocation.greedy_portfolio()
                allocation.update(allo)

        else:
            dfInvest = dfInvest.join(dfProtection)
            weights = self._calculate_weights(dfInvest,dfBenchmark,strat,solver,verbose)

            latestPrices = get_latest_prices(dfInvest)
            
            discAllocation = DiscreteAllocation(weights,latestPrices, total_portfolio_value=self.portfolioValue)
            allocation, leftover = discAllocation.greedy_portfolio()
            
        self._lastAllocation = allocation
        self._lastLeftover = leftover
        self._daily_portfolio_changes(date)
        
    def create_report(self) -> str:
        '''
        Create quantstats report for the model and benchmark. Generating metrics reports, plots, 
        and putting it all together in a tear sheet that will be saved as an HTML file.
        
        :return: filename where the output has been saved
        :type: str
        '''

        filename = '{}/ModelReport_{}.html'.format(os.getcwd(),strftime('%Y.%m.%d_%H%M%S',localtime()))
        returns = expected_returns.returns_from_prices(self.portfolioMetrics['Portfolio Value'])
        benchmark = expected_returns.returns_from_prices(self._dfBenchmark[self._dfBenchmark.index>=self.startDate]['SPX'])
        qs.reports.html(returns, benchmark,rf=self.riskFreeRate, title= 'Model Performance', output=filename)
        return filename