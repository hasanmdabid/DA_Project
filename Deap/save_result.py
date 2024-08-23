# This is the function to save the results.
# the inputs are: Method, modelname, epochs, batch_size, accuracy, f1score_macro, number of kerns, and filter size
# pylint: disable-all
def saveResultsCSV(label, aug_method, aug_factor, modelname, epochs, batch_size, test_acc, best_f1score_macro, avg_f1score, std_f1score, best_accuracy, avg_acc, std_acc, all_accuracies, all_f1_scores):
    import os.path
    from pathlib import Path
    from datetime import datetime
    
    path = './Deap/results/'
    if not os.path.exists(path):
        os.makedirs(path)

    fileString = './Deap/results/deap.csv'
    file = Path(fileString)

    now = datetime.now()

    if not file.exists():
        f = open(fileString, 'w')
        f.write('Finished on;label;aug_method;aug_factor;modelname;epochs;batch_size;test_acc;best_f1score_macro;avg_F1Score_macro;std_f1score_macro;best_accuracy;avg_acc;std_acc;all_accuracies;all_f1_scores\n')
        f.close()

    with open(fileString, "a") as f:
        all_accuracies_str = ','.join(map(str, all_accuracies))
        all_f1_scores_str = ','.join(map(str, all_f1_scores))
        f.write('{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}\n'.format(
            now, label, aug_method, aug_factor, modelname, epochs, batch_size, test_acc, 
            best_f1score_macro, avg_f1score, std_f1score, best_accuracy, avg_acc, std_acc, all_accuracies_str, all_f1_scores_str
        ))
        f.close()

    


