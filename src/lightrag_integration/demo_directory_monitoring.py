#!/usr/bin/env python3
"""
Directory Monitoring Demo

This script demonstrates the directory monitoring and batch processing
functionality implemented for task 6.
"""

import asyncio
import tempfile
from pathlib import Path
import time

from .component import LightRAGComponent
from .config.settings import LightRAGConfig


async def demo_directory_monitoring():
    """Demonstrate directory monitoring functionality."""
    
    print("🚀 LightRAG Directory Monitoring Demo")
    print("=" * 50)
    
    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temporary directory: {temp_dir}")
        
        # Create test configuration
        config = LightRAGConfig()
        config.papers_directory = str(Path(temp_dir) / "papers")
        config.knowledge_graph_path = str(Path(temp_dir) / "kg")
        config.vector_store_path = str(Path(temp_dir) / "vectors")
        config.cache_directory = str(Path(temp_dir) / "cache")
        
        # Create papers directory
        papers_dir = Path(config.papers_directory)
        papers_dir.mkdir(parents=True, exist_ok=True)
        print(f"📂 Created papers directory: {papers_dir}")
        
        # Initialize LightRAG component
        component = LightRAGComponent(config)
        
        try:
            print("\n🔧 Initializing LightRAG component...")
            await component.initialize()
            print("✅ Component initialized successfully")
            
            # Test 1: Get initial monitoring status
            print("\n📊 Getting initial monitoring status...")
            status = await component.get_monitoring_status()
            print(f"   Status: {status['status']}")
            print(f"   Papers directory: {status['papers_directory']}")
            print(f"   Directory exists: {status['papers_directory_exists']}")
            
            # Test 2: Force scan empty directory
            print("\n🔍 Performing force scan on empty directory...")
            scan_result = await component.force_directory_scan()
            print(f"   Files found: {scan_result['new_files_found']}")
            print(f"   Scan duration: {scan_result['scan_duration']:.3f}s")
            
            # Test 3: Create test PDF files
            print("\n📄 Creating test PDF files...")
            test_files = []
            for i in range(3):
                test_file = papers_dir / f"clinical_metabolomics_{i+1}.pdf"
                test_file.write_text(f"""
                Clinical Metabolomics Research Paper {i+1}
                
                Abstract: This paper discusses clinical metabolomics approaches
                for biomarker discovery and disease diagnosis.
                
                Keywords: metabolomics, biomarkers, clinical research
                """)
                test_files.append(str(test_file))
                print(f"   Created: {test_file.name}")
            
            # Test 4: Force scan with PDF files
            print("\n🔍 Performing force scan with PDF files...")
            scan_result = await component.force_directory_scan()
            print(f"   Files found: {scan_result['new_files_found']}")
            print(f"   Files: {[Path(f).name for f in scan_result['new_files']]}")
            print(f"   Scan duration: {scan_result['scan_duration']:.3f}s")
            
            # Test 5: Batch process files
            print("\n⚙️ Testing batch processing...")
            batch_result = await component.batch_process_files(test_files)
            print(f"   Total files: {batch_result['total_files']}")
            print(f"   Successful: {batch_result['successful']}")
            print(f"   Failed: {batch_result['failed']}")
            print(f"   Processing time: {batch_result['processing_time']:.3f}s")
            print(f"   Average time per file: {batch_result['average_time_per_file']:.3f}s")
            
            # Test 6: Get progress report
            print("\n📈 Getting progress report...")
            progress = await component.get_progress_report()
            print(f"   Total operations: {progress['total_operations']}")
            print(f"   Active operations: {progress['active_operations_count']}")
            print(f"   Completed operations: {progress['completed_operations_count']}")
            print(f"   Overall success rate: {progress['overall_success_rate']:.1f}%")
            
            # Test 7: Start directory monitoring
            print("\n🎯 Starting directory monitoring...")
            await component.start_directory_monitoring()
            
            status = await component.get_monitoring_status()
            print(f"   Monitoring status: {status['status']}")
            print(f"   Scan interval: {status['scan_interval']}s")
            print(f"   Batch size: {status['batch_size']}")
            
            # Test 8: Add new file while monitoring
            print("\n📄 Adding new file while monitoring is active...")
            new_file = papers_dir / "new_research_paper.pdf"
            new_file.write_text("""
            New Clinical Metabolomics Research
            
            This is a newly added research paper that should be detected
            by the directory monitoring system.
            """)
            print(f"   Created: {new_file.name}")
            
            # Wait for monitoring to detect the new file
            print("   Waiting for monitoring to detect new file...")
            await asyncio.sleep(2)
            
            # Check monitoring stats
            status = await component.get_monitoring_status()
            stats = status['stats']
            print(f"   Files detected: {stats['files_detected']}")
            print(f"   Files processed: {stats['files_processed']}")
            print(f"   Scan count: {stats['scan_count']}")
            
            # Test 9: Stop directory monitoring
            print("\n🛑 Stopping directory monitoring...")
            await component.stop_directory_monitoring()
            
            status = await component.get_monitoring_status()
            print(f"   Monitoring status: {status['status']}")
            
            # Test 10: Final progress report
            print("\n📊 Final progress report...")
            progress = await component.get_progress_report()
            print(f"   Total operations: {progress['total_operations']}")
            print(f"   Completed operations: {progress['completed_operations_count']}")
            print(f"   Failed operations: {progress['failed_operations_count']}")
            print(f"   Overall success rate: {progress['overall_success_rate']:.1f}%")
            
            print("\n✅ Directory monitoring demo completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Error during demo: {str(e)}")
            raise
        finally:
            print("\n🧹 Cleaning up...")
            await component.cleanup()
            print("✅ Cleanup completed")


async def demo_progress_tracking():
    """Demonstrate progress tracking functionality."""
    
    print("\n🎯 Progress Tracking Demo")
    print("=" * 30)
    
    from .ingestion.progress_tracker import ProgressTracker, OperationType
    
    # Create progress tracker
    config = {'max_history': 100, 'persist_to_file': False}
    tracker = ProgressTracker(config)
    
    try:
        await tracker.start()
        print("✅ Progress tracker started")
        
        # Start multiple operations
        print("\n📋 Starting multiple operations...")
        
        # Operation 1: File processing
        op1 = tracker.start_operation(
            operation_id="file_proc_1",
            operation_type=OperationType.FILE_PROCESSING,
            total_items=5,
            metadata={"file": "research_paper_1.pdf"}
        )
        print(f"   Started: {op1.operation_id} ({op1.operation_type.value})")
        
        # Operation 2: Batch processing
        op2 = tracker.start_operation(
            operation_id="batch_proc_1",
            operation_type=OperationType.BATCH_PROCESSING,
            total_items=10,
            metadata={"batch_size": 10}
        )
        print(f"   Started: {op2.operation_id} ({op2.operation_type.value})")
        
        # Simulate progress updates
        print("\n📈 Simulating progress updates...")
        for i in range(3):
            await asyncio.sleep(0.1)
            
            # Update file processing
            tracker.update_progress(
                operation_id="file_proc_1",
                completed_items=i + 1,
                current_step=f"Processing item {i + 1}"
            )
            
            # Update batch processing
            tracker.update_progress(
                operation_id="batch_proc_1",
                completed_items=(i + 1) * 2,
                current_step=f"Processing batch item {(i + 1) * 2}"
            )
            
            print(f"   Progress update {i + 1} completed")
        
        # Complete operations
        print("\n✅ Completing operations...")
        tracker.complete_operation("file_proc_1", success=True)
        tracker.complete_operation("batch_proc_1", success=True)
        
        # Get status report
        print("\n📊 Final status report...")
        report = tracker.get_status_report()
        print(f"   Total operations: {report.total_operations}")
        print(f"   Active operations: {len(report.active_operations)}")
        print(f"   Completed operations: {len(report.completed_operations)}")
        print(f"   Overall success rate: {report.overall_success_rate:.1f}%")
        
        # Show operation history
        print("\n📚 Operation history:")
        history = tracker.get_operation_history(limit=5)
        for op in history:
            duration = op.duration.total_seconds() if op.duration else 0
            print(f"   {op.operation_id}: {op.status.value} ({duration:.3f}s)")
        
    finally:
        await tracker.stop()
        print("✅ Progress tracker stopped")


async def main():
    """Run the complete demo."""
    print("🎬 LightRAG Directory Monitoring & Progress Tracking Demo")
    print("=" * 60)
    
    try:
        # Run directory monitoring demo
        await demo_directory_monitoring()
        
        # Run progress tracking demo
        await demo_progress_tracking()
        
        print("\n🎉 All demos completed successfully!")
        
    except Exception as e:
        print(f"\n💥 Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())