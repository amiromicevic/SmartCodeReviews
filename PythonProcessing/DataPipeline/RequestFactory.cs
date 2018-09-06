using Rally.RestApi;
using System.Configuration;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System;

namespace PythonProcessing.DataPipeline
{
	public class RallyRequestFactory
	{
		/// <summary>
		/// Creates a default request to Rally by offering more flexibility to client to inject parameters
		/// </summary>
		/// <param name="requestName"></param>
		/// <returns></returns>
		public Request CreateRequest(string requestName, List<string> valuesToFetch, string workspaceRef, string projectRef)
		{
			Request rallyRequest = new Request(requestName);
			rallyRequest.Workspace = workspaceRef;
			rallyRequest.Project = projectRef;
			rallyRequest.Fetch = valuesToFetch;
			return rallyRequest;
		}


		/// <summary>
		/// Creates a default request to Rally using the pre-configured parameters
		/// </summary>
		/// <param name="requestName"></param>
		/// <returns></returns>
		public Request CreateRequest(string requestName)
		{
			string workspaceRef, projectRef;
			List<string> valuesToFetch;

			try
			{
				workspaceRef = ConfigurationManager.AppSettings["WorkspaceRef"];
				projectRef = ConfigurationManager.AppSettings["ProjectRef"];
				valuesToFetch = ConfigurationManager.AppSettings["ValuesToFetch"].Split(",".ToCharArray()).ToList();
			}
			catch (Exception e)
			{
				throw new Exception("Configuration file setup is missing application critical keys.", e.InnerException);
			}

			Request rallyRequest = CreateRequest(requestName, valuesToFetch, workspaceRef, projectRef);
			return rallyRequest;
		}
	
	}
}
