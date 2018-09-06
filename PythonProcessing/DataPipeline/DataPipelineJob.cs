using PythonProcessing.DataLayer;
using Rally.RestApi;
using Rally.RestApi.Response;
using System;
using System.Collections.Generic;
using System.Configuration;
using System.Linq;

namespace PythonProcessing.DataPipeline
{
	public class DataPipelineJob : DataPipelineJobTemplate
	{
		private RallyRestApi _rallyApi;

		public override void Connect()
		{
			var login = ConfigurationManager.AppSettings["Login"];
			var password = ConfigurationManager.AppSettings["Password"];
			var rallyEndPoint = ConfigurationManager.AppSettings["RallyEndPoint"];

			_rallyApi = new RallyRestApi();
			_rallyApi.Authenticate(login, password, rallyEndPoint, proxy: null, allowSSO: false);
		}

		public override List<string> GetIterations()
		{
			var requestFactory = new RallyRequestFactory(); //rather simplified factory to fit the purpose
			var rallyRequest = requestFactory.CreateRequest("Iteration");
			QueryResult queryResults = _rallyApi.Query(rallyRequest);
			
			var iterations = new List<string>();

			foreach (var rallyItem in queryResults.Results.Reverse())
			{
				if (!iterations.Contains(rallyItem["Name"]))
				{
					iterations.Add(rallyItem["Name"]);
				}
			}

			return iterations;
		}

		public override List<object> GetStories(List<string> iterations)
		{
			List<object> stories = new List<object>();
			var resultSets = new List<QueryResult>();

			var requestFactory = new RallyRequestFactory(); 

			foreach (var iterationName in iterations)
			{			
				//get stories			
				var result = ExecuteRequest(requestFactory.CreateRequest("HierarchicalRequirement"), iterationName);
				resultSets.Add(result);

				//now get defects
				result = ExecuteRequest(requestFactory.CreateRequest("Defect"), iterationName);
				resultSets.Add(result);		

				foreach (var resultSet in resultSets)
				{
					foreach (var rallyItem in resultSet.Results)
					{
						if (rallyItem["Iteration"] != null && rallyItem["Iteration"]["Name"] == iterationName)
						{
							stories.Add(rallyItem);
						}
					}
				}
			}

			return stories;
		}

		public override List<string> GetCodeReviews(List<object> stories)
		{
			var codeReviews = new List<string>();

			foreach (var item in stories)
			{
				var _queryTaskResults = GetTasks(item);

				foreach (var rallyTask in _queryTaskResults.Results)
				{
					if (rallyTask["Name"] != null && IsThisCodeReviewTask(rallyTask["Name"]))
					{				
						var codeReview = rallyTask["Notes"] != null ? rallyTask["Notes"] : string.Empty;
						if (!string.IsNullOrEmpty(codeReview)) codeReviews.Add(codeReview);						
					}
				}
			}

			return codeReviews;
		}

		public override void UploadCodeReviewsToDb(List<string> codeReviews)
		{
			var destinationTable = "CodeReview";

			var columnMappings = new Dictionary<string, Type>()
			{
				{ "CodeReview", typeof(string)}
			};

			IDataLayer adoNetDb = new AdoNetDb();

			var dataToUpload = adoNetDb.PopulateDataTable(destinationTable, columnMappings, codeReviews);
			adoNetDb.ExecuteSQLBulkCopyInsert(destinationTable, columnMappings.Keys.ToList(), dataToUpload);
		}

		private QueryResult ExecuteRequest(Request rallyRequest, string iterationName)
		{
			var query = new Query("Iteration.Name", Query.Operator.Equals, iterationName);
			rallyRequest.Query = query;

			return _rallyApi.Query(rallyRequest);
		}

		private bool IsThisCodeReviewTask(string taskName)
		{
			return taskName.ToLower().Contains("code") && taskName.ToLower().Contains("review");
		}

		private QueryResult GetTasks(object item)
		{
			var tasksRequest = new Request(((dynamic)item)["Tasks"]);
			return _rallyApi.Query(tasksRequest);
		}
	}
}
