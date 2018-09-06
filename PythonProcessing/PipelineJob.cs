using PythonProcessing.DataPipeline;
using PythonProcessing.Processing;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PythonProcessing
{
	public class PipelineJob
	{
		public void Execute(IDataPipelineJob dataPipelineJob, IPythonJob pythonJob)
		{
			dataPipelineJob.Run();
			pythonJob.ProcessNewData();
		}
	}
}
