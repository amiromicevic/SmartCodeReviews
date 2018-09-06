using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PythonProcessing.DataPipeline
{
	public abstract class DataPipelineJobTemplate : IDataPipelineJob
	{
		public abstract void Connect();

		public abstract List<string> GetIterations();

		public abstract List<object> GetStories(List<string> iterations);

		public abstract List<string> GetCodeReviews(List<object> stories);

		public abstract void UploadCodeReviewsToDb(List<string> codeReviews);

		public void Run()
		{
			Connect();
			List<string> iterations = GetIterations();
			List<object> stories = GetStories(iterations);
			List<string> codeReviews = GetCodeReviews(stories);
			UploadCodeReviewsToDb(codeReviews);
		}
	}
}
