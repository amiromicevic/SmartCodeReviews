using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PythonProcessing.DataPipeline
{
	public interface IDataPipelineJob
	{
		void Connect();

		List<string> GetIterations();

		List<object> GetStories(List<string> iterations);

		List<string> GetCodeReviews(List<object> stories);

		void UploadCodeReviewsToDb(List<string> codeReviews);

		void Run();
	}
}
