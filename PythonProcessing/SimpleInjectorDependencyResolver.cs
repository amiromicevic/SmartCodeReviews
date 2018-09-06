using PythonProcessing.DataLayer;
using PythonProcessing.DataPipeline;
using PythonProcessing.Processing;
using SimpleInjector;

namespace PythonProcessing
{
	public static class SimpleInjectorDependencyResolver
	{
		public static Container SetContainer()
		{		
			var container = new Container();
			container.Register<IDataLayer, AdoNetDb>();
			container.Register<IPythonJob, PythonJob>();
			container.Register<IDataPipelineJob, DataPipelineJob>();

			return container;
		}		
	}
}
