#ifndef LUMINOCUGP_PREFIX_CUH
#define LUMINOCUGP_PREFIX_CUH

#include <stack>
#include <iostream>
#include <random>
#include <cmath>
#include <string>
#include <sstream>
#include <utility>

/**
 * weights in finding cutting point
 */
#define FUNCTION_WEIGHTS 0.9
#define TERMINAL_WEIGHTS 0.1

namespace cusr {
    namespace program {

        using namespace std;

        typedef enum NodeType {
            VAR,   // variable
            CONST, // constant
            UFUNC, // unary function
            BFUNC  // binary function
        } ntype_t;

        typedef enum Function {
            ADD, // arity: 2, return a + b
            SUB, // arity: 2, return a - b
            MUL, // arity: 2, return a * b
            DIV, // arity: 2, if (b == 0) { b = DELTA } return a / b
            TAN, // arity: 1, return tan a
            SIN, // arity: 1, return sin a
            COS, // arity: 1, return cos a
            LOG, // arity: 1, return log a
            MAX, // arity: 2, if (a > b) { return a } return b
            MIN, // arity: 2, if (a < b) { return a } return b
            INV,  // arity: 1, if (a == 0) { a = DELTA } return 1 / a
            POW,
            EXP,
            SQU,
            CUB,
            LOG1P
        } func_t;

        typedef enum InitMethod {
            half_and_half,
            growth,
            full
        } init_t;

        struct Node {
            ntype_t node_type;  // type of the node
            float constant;     // value of constant
            int variable;       // the number of variable (e.g., 0 refers to x0; 1 refers to x1).
            func_t function;    // type of function
        };

        typedef std::vector<Node> prefix_t;

        /**
         * random integer
         *
         * @param loBound
         * @param upBound
         * @return
         */
        void initialize_random_engine(unsigned seed = 0);
        /**
         * Random integer using std::mt19937 with fixed seed.
         * @param loBound lower bound
         * @param upBound upper bound
         * @return random integer
         */
        int gen_rand_int(int loBound, int upBound);

        /**
         * Random float using std::mt19937 with fixed seed.
         * @param loBound lower bound
         * @param upBound upper bound
         * @return random float
         */
        float gen_rand_float(float loBound, float upBound);

        /**
         * returns the depth of a expression tree represented by a prefix
         *
         * @param prefix
         * @return
         */
        int get_depth_of_prefix(prefix_t &prefix);

        /**
         * convert prefix to infix (for log only)
         *
         * @param prefix
         * @return
         */
        string prefix_to_infix(prefix_t &prefix);

        /**
         * get the index of a subtree
         * returns the index {start pos, end_pos + 1} of the subtree
         * @param prefix
         * @param allow_terminal
         * @return
         */
        pair<int, int> rand_subtree_index_roulette(prefix_t &prefix, bool allow_terminal);

        /**
         * generate random constant terminal
         *
         * @param node
         * @param range
         */
        void rand_constant(Node &node, pair<float, float> &range);

        /**
         * generate random variable terminal
         *
         * @param node
         * @param variable_num
         */
        void rand_variable(Node &node, int variable_num);

        /**
         * generate random terminal (variable or constant)
         *
         * @param node
         * @param range
         * @param variable_num
         * @param p_constant probability to generate a constant
         */
        void rand_terminal(Node &node, pair<float, float> &range, int variable_num, float p_constant);

        /**
         * generate random terminal (variable or constant), the probability to generate a constant is 50%
         *
         * @param node
         * @param range
         * @param variable_num
         */
        void rand_terminal(Node &node, pair<float, float> &range, int variable_num);

        /**
         * generate random function
         *
         * @param node
         * @param function_set
         */
        void rand_function(Node &node, vector<Function> &function_set);

        struct TreeNode {
            TreeNode() : left(nullptr), right(nullptr) {}

            Node node;
            TreeNode *left;
            TreeNode *right;
        };

        /**
         * full initialization
         *
         * @param depth
         * @param range
         * @param func_set
         * @param variable_num
         * @return
         */
        TreeNode *
        gen_full_init_tree(int depth, pair<float, float> &range, vector<Function> &func_set, int variable_num);

        /**
         * growth initialization
         *
         * @param depth
         * @param range
         * @param func_set
         * @param variable_num
         * @return
         */
        TreeNode *
        gen_growth_init_tree(int depth, pair<float, float> &range, vector<Function> &func_set, int variable_num);

        /**
         * convert an expression tree to a prefix
         *
         * @param prefix
         * @param tree_node
         */
        void get_init_prefix(prefix_t &prefix, TreeNode *tree_node);

        /**
         * convert a prefix to string (for printing it on the console)
         *
         * std::cout << prefix_to_string(prefix) << std::endl;
         * @param prefix
         * @return
         */
        string prefix_to_string(prefix_t &prefix);

        /**
         * find rand cutting point by roulette
         *
         * @param prefix
         * @param allow_terminal
         * @return
         */
        int rand_roulette_pos(prefix_t &prefix, bool allow_terminal);

        /**
         * get the index of subtree
         * @param prefix
         * @param start_pos
         * @return
         */
        pair<int, int> get_subtree_index(prefix_t &prefix, int start_pos);

        /**
         * probability to gen random constant
         * @param p_const
         */
        void set_constant_prob(float p_const);

        /**
         * set seed using times
         * gen random number n times then change the seed of random engine
         * @param n
         */
        void set_seed_using_times(int n);
    }
}
#endif //LUMINOCUGP_PREFIX_CUH
