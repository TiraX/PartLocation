import os
import json
import time
from pathlib import Path
from typing import List, Callable, Any, Dict, Optional
from multiprocessing import Process, Queue, Manager, cpu_count
from dataclasses import dataclass, asdict
from enum import Enum


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """
    Represents a single task to be executed.
    
    Attributes:
        task_id: Unique identifier for the task
        data: Task-specific data to be processed
        status: Current status of the task
        error_msg: Error message if task failed
    """
    task_id: str
    data: Any
    status: TaskStatus = TaskStatus.PENDING
    error_msg: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert task to dictionary for serialization"""
        return {
            'task_id': self.task_id,
            'data': self.data,
            'status': self.status.value,
            'error_msg': self.error_msg
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Task':
        """Create task from dictionary"""
        return cls(
            task_id=data['task_id'],
            data=data['data'],
            status=TaskStatus(data['status']),
            error_msg=data.get('error_msg')
        )


class ParallelTasks:
    """
    A multi-process framework for executing tasks in parallel with checkpoint/resume support.
    
    Features:
    - Dynamic task dispatching to maximize resource utilization
    - Checkpoint/resume capability to continue from where it left off
    - Automatic worker management
    """
    
    def __init__(
        self,
        task_func: Callable[[Any], Any],
        checkpoint_file: Optional[str],
        num_processes: Optional[int] = None
    ):
        """
        Initialize the parallel tasks framework.
        
        Args:
            task_func: Function to execute for each task. Should accept task data and return result.
            checkpoint_file: Base path for checkpoint files (will append worker_id)
            num_processes: Number of worker processes. If None, uses cpu_count().
        """
        self.task_func = task_func
        self.checkpoint_file = checkpoint_file
        self.num_processes = num_processes or cpu_count()
        self.tasks: List[Task] = []
        self.completed_task_ids: set = set()
        
    def create_tasks(self, task_generator: Callable[[], List[Task]]) -> None:
        """
        Create tasks using a task generator function.
        
        Args:
            task_generator: Function that returns a list of Task objects
        """
        self.tasks = task_generator()
        print(f'Created {len(self.tasks)} tasks')
        
    def _get_checkpoint_path(self, worker_id: int) -> str:
        """Get checkpoint file path for a specific worker"""
        if not self.checkpoint_file:
            return None
        
        base_path = Path(self.checkpoint_file)
        return str(base_path.parent / f"{base_path.stem}_worker{worker_id}{base_path.suffix}")
    
    def load_checkpoint(self) -> None:
        """Load all checkpoint files from all workers"""
        if not self.checkpoint_file:
            print('No checkpoint file configured, starting fresh')
            return
        
        self.completed_task_ids = set()
        loaded_count = 0
        
        # Load checkpoints from all workers
        for worker_id in range(self.num_processes):
            checkpoint_path = self._get_checkpoint_path(worker_id)
            if not os.path.exists(checkpoint_path):
                continue
            
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                
                completed_tasks = checkpoint_data.get('completed_tasks', [])
                self.completed_task_ids.update(completed_tasks)
                loaded_count += len(completed_tasks)
                
            except Exception as e:
                print(f'Error loading checkpoint from worker {worker_id}: {e}')
        
        if loaded_count > 0:
            print(f'Loaded checkpoint: {len(self.completed_task_ids)} tasks already completed')
    
    @staticmethod
    def _save_worker_checkpoint(checkpoint_path: str, completed_task_ids: List[str]) -> None:
        """
        Save checkpoint for a specific worker.
        
        Args:
            checkpoint_path: Path to checkpoint file
            completed_task_ids: List of completed task IDs
        """
        if not checkpoint_path:
            return
        
        try:
            # Ensure directory exists
            checkpoint_dir = os.path.dirname(checkpoint_path)
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_data = {
                'completed_tasks': completed_task_ids,
                'last_update': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2)
                
        except Exception as e:
            print(f'Error saving checkpoint: {e}')
    
    def get_pending_tasks(self) -> List[Task]:
        """
        Get list of tasks that haven't been completed yet.
        
        Returns:
            List of pending tasks
        """
        pending_tasks = [
            task for task in self.tasks 
            if task.task_id not in self.completed_task_ids
        ]
        return pending_tasks
    
    @staticmethod
    def _worker_process(
        worker_id: int,
        assigned_tasks: List[Dict],
        result_queue: Queue,
        task_func: Callable[[Any], Any],
        checkpoint_path: Optional[str]
    ) -> None:
        """
        Worker process that executes pre-assigned tasks.
        
        Args:
            worker_id: Unique identifier for this worker
            assigned_tasks: List of tasks assigned to this worker
            result_queue: Queue to put results into
            task_func: Function to execute for each task
            checkpoint_path: Path to this worker's checkpoint file
        """
        # print(f'Worker {worker_id}: Started with {len(assigned_tasks)} tasks')
        
        completed_task_ids = []
        
        for task_dict in assigned_tasks:
            try:
                task = Task.from_dict(task_dict)
                print(f'Worker {worker_id}: Processing task {task.task_id}/{len(assigned_tasks)}')
                
                try:
                    # Execute the task
                    start_time = time.time()
                    result = task_func(task.data)
                    elapsed_time = time.time() - start_time
                    
                    task.status = TaskStatus.COMPLETED
                    completed_task_ids.append(task.task_id)
                    print(f'Worker {worker_id}: Completed task {task.task_id} in {elapsed_time:.2f}s')
                    
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error_msg = str(e)
                    print(f'Worker {worker_id}: Task {task.task_id} failed: {e}')
                
                # Put result back
                result_queue.put(task.to_dict())
                
                # Save checkpoint after each task
                if checkpoint_path and task.status == TaskStatus.COMPLETED:
                    ParallelTasks._save_worker_checkpoint(checkpoint_path, completed_task_ids)
                
            except Exception as e:
                print(f'Worker {worker_id}: Error processing task: {e}')
                continue
        
        print(f'Worker {worker_id}: Completed all assigned tasks')
    
    def run(self) -> Dict[str, Any]:
        """
        Execute all pending tasks using multiple worker processes with pre-allocated tasks.
        
        Returns:
            Dictionary containing execution statistics
        """
        # Load checkpoint to skip completed tasks
        self.load_checkpoint()
        
        # Get pending tasks
        pending_tasks = self.get_pending_tasks()
        total_tasks = len(self.tasks)
        pending_count = len(pending_tasks)
        
        print(f'\n=== Task Execution Summary ===')
        print(f'Total tasks: {total_tasks}')
        print(f'Already completed: {len(self.completed_task_ids)}')
        print(f'Pending tasks: {pending_count}')
        print(f'Worker processes: {self.num_processes}')
        print(f'================================\n')
        
        if pending_count == 0:
            print('All tasks already completed!')
            return {
                'total_tasks': total_tasks,
                'completed': len(self.completed_task_ids),
                'failed': 0,
                'skipped': 0
            }
        
        # Pre-allocate tasks to workers
        tasks_per_worker = [[] for _ in range(self.num_processes)]
        for idx, task in enumerate(pending_tasks):
            worker_id = idx % self.num_processes
            tasks_per_worker[worker_id].append(task.to_dict())
        
        # Print task allocation
        for worker_id, tasks in enumerate(tasks_per_worker):
            print(f'Worker {worker_id}: Assigned {len(tasks)} tasks')
        
        # Create result queue
        manager = Manager()
        result_queue = manager.Queue()
        
        # Create all worker processes first
        workers = []
        for worker_id in range(self.num_processes):
            if not tasks_per_worker[worker_id]:
                continue  # Skip workers with no tasks
            
            checkpoint_path = self._get_checkpoint_path(worker_id)
            worker = Process(
                target=self._worker_process,
                args=(worker_id, tasks_per_worker[worker_id], result_queue, self.task_func, checkpoint_path)
            )
            workers.append(worker)
        
        # Start all workers at once
        for worker in workers:
            worker.start()
        
        # Collect results
        completed_count = 0
        failed_count = 0
        start_time = time.time()
        
        while completed_count + failed_count < pending_count:
            try:
                # Get result from queue
                result_dict = result_queue.get(timeout=1)
                task = Task.from_dict(result_dict)
                
                if task.status == TaskStatus.COMPLETED:
                    completed_count += 1
                    processed = completed_count + failed_count
                    percentage = (processed / pending_count) * 100
                    print(f'Progress: {processed}/{pending_count} ({percentage:.1f}%)')
                    
                elif task.status == TaskStatus.FAILED:
                    failed_count += 1
                    processed = completed_count + failed_count
                    percentage = (processed / pending_count) * 100
                    print(f'Task {task.task_id} failed: {task.error_msg}')
                    print(f'Progress: {processed}/{pending_count} ({percentage:.1f}%)')
                    
            except Exception as e:
                # Queue timeout
                continue
        
        # Wait for all workers to finish
        for worker in workers:
            worker.join()
        
        elapsed_time = time.time() - start_time
        
        print(f'\n=== Execution Complete ===')
        print(f'Total time: {elapsed_time:.2f}s')
        print(f'Completed: {completed_count}')
        print(f'Failed: {failed_count}')
        print(f'==========================\n')
        
        return {
            'total_tasks': total_tasks,
            'completed': len(self.completed_task_ids) + completed_count,
            'failed': failed_count,
            'skipped': len(self.completed_task_ids),
            'elapsed_time': elapsed_time
        }


# Example usage
def example_task_function(data: Dict) -> Any:
    """
    Example task function that processes task data.
    
    Args:
        data: Task-specific data
        
    Returns:
        Processing result
    """
    # Simulate some work
    task_name = data.get('name', 'unknown')
    duration = data.get('duration', 1)
    
    print(f'Processing {task_name}...')
    time.sleep(duration)
    
    return f'Completed {task_name}'


def example_task_generator() -> List[Task]:
    """
    Example task generator that creates a list of tasks.
    
    Returns:
        List of Task objects
    """
    tasks = []
    for i in range(20):
        task = Task(
            task_id=f'task_{i:03d}',
            data={
                'name': f'Task {i}',
                'duration': 0.5 + (i % 3) * 0.5  # Variable duration
            }
        )
        tasks.append(task)
    
    return tasks


def main():
    """Example main function demonstrating the framework usage"""
    # Create parallel tasks framework
    parallel_tasks = ParallelTasks(
        task_func=example_task_function,
        checkpoint_file='checkpoint.json',
        num_processes=4
    )
    
    # Create tasks
    parallel_tasks.create_tasks(example_task_generator)
    
    # Run tasks
    stats = parallel_tasks.run()
    
    print(f'Final statistics: {stats}')


if __name__ == '__main__':
    main()
