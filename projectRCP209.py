import pandas as pd
###data from CFM
repPath = "C:\\Users\\jerpetit\\Desktop\\master\\TRIED\\S1\\RCP209\\project\\data\\"
outputPath = repPath + "challenge_output_data_training_file_volatility_prediction_in_financial_markets.csv"
testPath = repPath + "training_input.csv"
trainPath = repPath + "testing_input.csv"
train = pd.read_csv(trainPath)
test = pd.read_csv(testPath)
output = pd.read_csv(outputPath)

###print data info
print(train.columns)
print(test.columns)
print(output.columns)