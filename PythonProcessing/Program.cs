using PythonProcessing.Processing;
using SimpleInjector;
using PythonProcessing.DataPipeline;

namespace PythonProcessing
{
	class Program
	{
		static void Main(string[] args)
		{
			Container container = SimpleInjectorDependencyResolver.SetContainer();
			container.Verify();

			IDataPipelineJob dataPipelineJob = container.GetInstance<IDataPipelineJob>();
			IPythonJob pythonJob = container.GetInstance<IPythonJob>();
			
			PipelineJob pipeline = new PipelineJob();
			pipeline.Execute(dataPipelineJob, pythonJob);
		}
	}
}
