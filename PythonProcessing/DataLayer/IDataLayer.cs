using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PythonProcessing.DataLayer
{
	public interface IDataLayer
	{
		void ExecuteSQLBulkCopyInsert(string destinationTable, List<string> columnMappings, DataTable dtRowsToInsert);
		DataTable PopulateDataTable(string destinationTable, Dictionary<string, Type> columnMappings, List<string> content);

	}
}
