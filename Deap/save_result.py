# This is the function to save the results.
# the inputs are: Method, modelname, epochs, batch_size, accuracy, average_fscore_macro, number of kerns, and filter size

def saveResultsCSV(method, modelname, epochs, batch_size, accuracy, average_fscore_macro, average_fscore_weighted,
                   nKerns, filterSizes):
    
    import os.path
    from pathlib import Path
    from datetime import datetime
    path = './results/'
    if not os.path.exists(path):
        os.makedirs(path)

    fileString = './results/results.csv'
    file = Path(fileString)

    now = datetime.now()

    if not file.exists():
        f = open(fileString, 'w')
        f.write(
            'Finished on; Modelname; Epochs; Batch_Size; accuracy; Average_fscore_Macro; Average_fscores_weighted; nKerns; filterSize\n')
        f.close()
    with open(fileString, "a") as f:
        f.write(
            '{};{};{};{};{};{};{};{};{}:{}\n'.format(now, method, modelname, epochs, batch_size, accuracy, average_fscore_macro,
                                                  average_fscore_weighted, nKerns, filterSizes))
    f.close()
    


