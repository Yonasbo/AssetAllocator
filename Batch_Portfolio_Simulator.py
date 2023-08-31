from AllocatorClass import EfficientFrontierAllocator, DataLoader
import sys


def main():
    

    
    df_protection, df_benchmark, df_invest, df_risk = DataLoader(handlingNa=4)
    
    t= sys.argv[1] if len(sys.argv)>1 else '2007/01/01'
    
    
    days = int(sys.argv[2]) if len(sys.argv)>2 else -1
    Allocator = EfficientFrontierAllocator(dfProtection=df_protection,dfInvest=df_invest,
                                        dfBenchmark=df_benchmark,dfRisk=df_risk,startDate=t,runtime=days)

    timeframe = Allocator.timeframe
    year = 0
    for time in timeframe[1:]:
        if year!=time.year:
            year = time.year
            print('Currently evaluating {}.'.format(year))
        Allocator.get_new_portfolio_value(time)
        
        Allocator.calculate_next_allocation(time)

    output = Allocator.create_report()
    print(f'Model/Benchmark report has been saved at: {output}')

if __name__ == "__main__":
    main()

