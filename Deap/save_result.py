# This is the function to save the results.
# the inputs are: Method, modelname, epochs, batch_size, accuracy, f1score_macro, number of kerns, and filter size

def saveResultsCSV(aug_method, aug_factor, modelname, epochs, batch_size, train_acc, test_acc, best_f1score_macro, avg_f1score, std_f1score, best_accuracy, avg_acc, std_acc):
    
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
            'Finished on;aug_method;aug_factor;modelname;epochs;batch_size;train_acc;test_acc;best_f1score_macro;avg_F1Score_macro;std_f1score_macro;best_accuracy;avg_acc;std_acc\n')
        f.close()
    with open(fileString, "a") as f:
        f.write(
            '{};{};{};{};{};{};{};{};{};{};{};{};{};{}\n'.format(now, aug_method, aug_factor, modelname, epochs, batch_size, train_acc, test_acc, best_f1score_macro, avg_f1score, std_f1score, best_accuracy, avg_acc, std_acc))
    f.close()
    


