#ifndef LUMINOCUGP_REGRESSION_CUH
#define LUMINOCUGP_REGRESSION_CUH

#include <algorithm>
#include <sstream>
#include <unordered_set>
#include <utility>
#include "program.cuh"
#include "fit_eval.cuh"

namespace cusr {

    using namespace std;
    using namespace program;
    using namespace fit;

    /**
     * Symbolic regression engine.
     *
     * => begin
     *     do_population_init(CPU)
     *
     *     while best fitness > stopping criteria:
     *         selection(CPU)
     *         mutation(CPU)
     *         evaluation(CPU/GPU)
     * => end
     */
    class RegressionEngine {
    public:

        int population_size = 1000;
        int generations = 200;
        int tournament_size = 20;
        float stopping_criteria = 0.0;

        RegressionEngine() = default;

        ~RegressionEngine();

        /**
         * constant range
         * const_range.first  : lower bound
         * const_range.second : upper bound
         */
        pair<float, float> const_range = {-1.0, 1.0};

        /**
         * range of depth for program during initialization
         * init_depth.first  : lower bound
         * init_depth.second : upper bound
         */
        pair<int, int> init_depth = {4, 8};

        /**
         * init method of population
         *
         * init_t::full
         * init_t::growth
         * init_t::half_and_half
         */
        InitMethod init_method = InitMethod::half_and_half;

        /**
         * function set
         * =======================
         * function name | arity *
         * =======================
         * func_t::ADD   |  2    *
         * func_t::SUB   |  2    *
         * func_t::MUL   |  2    *
         * func_t::DIV   |  2    *
         * func_t::TAN   |  1    *
         * func_t::SIN   |  1    *
         * func_t::COS   |  1    *
         * func_t::LOG   |  1    *
         * func_t::INV   |  1    *
         * func_t::MAX   |  2    *
         * func_t::MIN   |  2    *
         * =======================
         */
        vector<Function> function_set = {
                Function::ADD, Function::SUB, Function::MUL,
                Function::DIV, Function::SIN, Function::COS,
                Function::TAN, Function::LOG, Function::INV
        };

        /**
         * metric type
         * default: metric_t::mean_absolute_error
         *
         * metric_t::mean_absolute_error
         * metric_t::mean_square_error
         * metric_t::root_mean_square_error
         */
        Metric metric = Metric::mean_absolute_error;

        /**
         * if or not the engine restrict the max depth of the program
         */
        bool restrict_depth = true;

        /**
         * depth restriction for a program, less than 20 is recommend
         * since GPU restrict the max length of the prefix is 2048, too large for this parameter may lead to the overflow
         */
        int max_program_depth = 10;

        /**
         * use only in selection
         * select_criterion = metric_loss + length * parsimony_coefficient
         */
        float parsimony_coefficient = 0;

        /**
         * probability to perform various mutations
         */
        float p_crossover = 0.9;
        float p_subtree_mutation = 0.01;
        float p_hoist_mutation = 0.01;
        float p_point_mutation = 0.01;
        float p_point_replace = 0.05;

        /**
         * probability to generate a random constant
         */
         float p_constant = 0.2;

        /**
         * use GPU acceleration
         */
        bool use_gpu = false;

        /**
         * fit dataset and training
         *
         * @param dataset
         * @param label
         */
        void fit(vector<vector<float>> &dataset, vector<float> &label);

        /**
         * predict
         * @param dataset
         * @return
         */
        // float predict(vector<float> dataset);

        /**
         * the best program with the best fitness in the last gen
         */
        Program best_program;

        /**
         * list of best program with the best fitness in each gen
         */
        vector<Program> best_program_in_each_gen;

        /**
         * iteration time
         */
        float regress_time_in_sec;

        /**
         * n_hall_of_fame
         */
        int n_hall_of_fame = 100;

        /**
         * n_components
         */
        int n_components = 10;

        /**
         * list of n_components number of programs for the symbolic transformer
         */
        vector<Program> components;

        /**
         * fit and train a symbolic transformer
         * 
         * @param dataset
         * @param label
         * @param corr 'pearson' or 'spearman'
         */
        void fit(vector<vector<float>> &dataset, vector<float> &label, string corr);

        /**
         * transform input dataset and generate n_components new features
         */
        void transform(vector<vector<float>> &dataset, vector<vector<float>> &new_dataset);

    private:

        GPUDataset device_dataset;
        vector<Program> population;
        vector<vector<float>> dataset;
        vector<float> label;

        int variable_nums;
        int max_length_in_population = 0;
        int max_depth_in_population = 0;

        void do_gpu_init();

        void do_fit_init();

        void do_population_init();

        Program do_mutation(Program &program);

        void gen_next_generation();

        void update_population_attributes();

        void calculate_population_fitness_cpu();

        void calculate_population_fitness_gpu();
    };

    /**
     * calculate correlation matrix of n vectors
     * 
     * @param data 
     * @param corr_matrix
     */
    void cal_corr_matrix(const vector<vector<float>> &data, vector<vector<float>> &corr_matrix);
}
#endif //LUMINOCUGP_REGRESSION_CUH
