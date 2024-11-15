import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.LinearRegression;
import weka.core.DenseInstance;
import weka.core.SerializationHelper;

public class SalesPrediction {
    public static void main(String[] args) throws Exception {
        // Suppress native BLAS and LAPACK warnings
        System.setProperty("com.github.fommil.netlib.BLAS", "false");
        System.setProperty("com.github.fommil.netlib.LAPACK", "false");
        System.err.close();  // This will close the error stream to suppress warnings

        // Load the dataset
        DataSource source = new DataSource("./data.csv");
        Instances dataset = source.getDataSet();

        // Print the dataset to verify its structure
        System.out.println("Dataset structure and values:");
        System.out.println(dataset);

        // Set the target attribute (class) as "Revenue"
        dataset.setClassIndex(dataset.attribute("Revenue").index());

        // Build the Linear Regression model
        LinearRegression model = new LinearRegression();
        model.buildClassifier(dataset);

        // Save the model to a file
        SerializationHelper.write("./linear_regression_model.model", model);
        System.out.println("Model saved successfully to linear_regression_model.model");

        // Print model summary
        System.out.println(model);

        // Predict revenue with a 10% increase in advertising cost for each month
        for (int i = 0; i < dataset.numInstances(); i++) {
            double currentCost = dataset.instance(i).value(dataset.attribute("Advertising_Cost"));
            double increasedCost = currentCost * 1.3;  // Increase cost by 10%

            // Create a new instance for prediction
            DenseInstance newInstance = new DenseInstance(dataset.numAttributes());
            newInstance.setDataset(dataset);

            // Copy over the values for Year, Month, and set increased Advertising_Cost
            newInstance.setValue(dataset.attribute("Year"), dataset.instance(i).value(dataset.attribute("Year")));
            newInstance.setValue(dataset.attribute("Month"), dataset.instance(i).value(dataset.attribute("Month")));
            newInstance.setValue(dataset.attribute("Advertising_Cost"), increasedCost);

            // Predict revenue for the increased advertising cost
            double predictedRevenue = model.classifyInstance(newInstance);
            System.out.printf("Predicted Revenue for month %.0f with 30%% increase in advertising cost (%.2f): %.2f%n",
                    dataset.instance(i).value(dataset.attribute("Month")),
                    increasedCost,
                    predictedRevenue);
        }
    }
}
