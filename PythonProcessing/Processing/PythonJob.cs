using IronPython.Hosting;
using Microsoft.Scripting.Hosting;
using System;
using System.Configuration;

namespace PythonProcessing.Processing
{
	public class PythonJob : IPythonJob
	{
		public void ProcessNewData()
		{
			//https://loosexaml.wordpress.com/2011/09/16/calling-ironpython-from-c/
			//https://stackoverflow.com/questions/28866916/call-a-python-function-with-multiple-return-values-from-c-sharp

			var engine = Python.CreateEngine(); 
			dynamic scope = engine.CreateScope();

			var pythonSource = ConfigurationManager.AppSettings["ProcessCodeReviewsPythonScript"];		
			ScriptSource source = engine.CreateScriptSourceFromFile(pythonSource); 

			object result = source.Execute(scope);		
			var x = scope.run();

			Console.ReadLine();
		}
	}
}
