# This is the function to save the results.
# the inputs are: Method, modelname, epochs, batch_size, accuracy, average_fscore_macro, number of kerns, and filter size

def saveResultsCSV(method, Aug_factor, modelname, epochs, batch_size, train_acc, test_acc, average_fscore_macro):
    
    import os.path
    from pathlib import Path
    from datetime import datetime
    path = './Deap/results/'
    if not os.path.exists(path):
        os.makedirs(path)

    fileString = './Deap/results/results.csv'
    file = Path(fileString)

    now = datetime.now()

    if not file.exists():
        f = open(fileString, 'w')
        f.write(
            'Finished on; Method; Aug_factor; Modelname; Epochs; Batch_Size; train_accuracy; test_accuracy; Average_fscore_macro;\n')
        f.close()
    with open(fileString, "a") as f:
        f.write(
            '{};{};{};{};{};{};{};{};{}\n'.format(now, method, Aug_factor, modelname, epochs, batch_size, train_acc, test_acc, average_fscore_macro))
    f.close()
    


