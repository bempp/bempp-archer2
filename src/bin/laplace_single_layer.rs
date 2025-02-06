//! This example benchmarks the time for evaluating the Laplace single layer operator
use std::time::Instant;

use bempp::boundary_assemblers::BoundaryAssemblerOptions;
use clap::Parser;
use mpi::traits::{Communicator, CommunicatorCollectives};
use ndelement::{ciarlet::LagrangeElementFamily, types::ReferenceCellType};
use rlst::{zero_element, AsApply, OperatorBase};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Number of physical cores per node
    #[arg(long, default_value_t = 128)]
    cores_per_node: usize,
    /// Number of tasks per node. Each task is one process
    #[arg(long, default_value_t = 32)]
    tasks_per_node: usize,
    /// Number of threads per task
    #[arg(long, default_value_t = 4)]
    threads_per_task: usize,
    /// Sphere refinement level r. Total number of elements: 8 * 4^r
    #[arg(long, default_value_t = 9)]
    refinement_level: usize,
    /// Use dense evaluation instead of FMM
    #[arg(long, default_value_t = false)]
    dense_evaluation: bool,
    /// The local tree depth for the FMM
    #[arg(long, default_value_t = 4)]
    local_tree_depth: usize,
    /// The global tree depth for the FMM
    #[arg(long, default_value_t = 4)]
    global_tree_depth: usize,
    /// FMM epansion order
    #[arg(long, default_value_t = 6)]
    expansion_order: usize,
}

// Define possible parameters with clap

fn main() {
    let args = Args::parse();

    // Initialise Rayon threading
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads_per_task)
        .build_global()
        .unwrap();

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    let now = Instant::now();
    let grid = bempp::shapes::regular_sphere::<f64, _>(args.refinement_level as u32, 1, &world);
    let elapsed = now.elapsed();

    if rank == 0 {
        println!("Grid generated in {} ms", elapsed.as_millis());
    }

    let quad_degree = 6;
    // Get the number of cells in the grid.

    if rank == 0 {
        println!("Instantiating function space.");
    }
    let now = Instant::now();
    let space = bempp::function::FunctionSpace::new(
        &grid,
        &LagrangeElementFamily::<f64>::new(1, ndelement::types::Continuity::Standard),
    );
    let elapsed = now.elapsed();

    world.barrier();

    if rank == 0 {
        println!("Function space generated in {} seconds", elapsed.as_secs());
    }

    let mut options = BoundaryAssemblerOptions::default();
    options.set_regular_quadrature_degree(ReferenceCellType::Triangle, quad_degree);

    let qrule = options.get_regular_quadrature_rule(ReferenceCellType::Triangle);

    // We now have to get all the points from the grid. We do this by iterating through all cells.

    // let kernel_evaluator =
    //     bempp::greens_function_evaluators::dense_evaluator::DenseEvaluator::from_spaces(
    //         &space,
    //         &space,
    //         green_kernels::types::GreenKernelEvalType::Value,
    //         true,
    //         Laplace3dKernel::new(),
    //         &qrule.points,
    //     );

    println!("Instantiating kifmm evaluator");
    let now = Instant::now();
    let kifmm_evaluator =
        bempp::greens_function_evaluators::kifmm_evaluator::KiFmmEvaluator::from_spaces(
            &space,
            &space,
            green_kernels::types::GreenKernelEvalType::Value,
            args.local_tree_depth,
            args.global_tree_depth,
            args.expansion_order,
            &qrule.points,
        );
    let elapsed = now.elapsed();

    if rank == 0 {
        println!("kifmm evaluator generated in {} seconds", elapsed.as_secs());
    }

    println!("Instantiating Laplace evaluator");
    let now = Instant::now();
    let laplace_evaluator =
        bempp::laplace::evaluator::single_layer(&space, &space, kifmm_evaluator.r(), &options);
    let elapsed = now.elapsed();

    if rank == 0 {
        println!(
            "Laplace evaluator generated in {} seconds",
            elapsed.as_secs()
        );
    }

    let mut x = zero_element(laplace_evaluator.domain());

    x.view_mut()
        .local_mut()
        .fill_from_seed_equally_distributed(rank as usize);

    println!("Apply the evalutor.");
    let now = Instant::now();
    let _res = laplace_evaluator.apply(x.r());
    let elapsed = now.elapsed();

    if rank == 0 {
        println!("Operator applied in {} seconds", elapsed.as_secs());
    }

    println!("Run successfully completed.");
}
