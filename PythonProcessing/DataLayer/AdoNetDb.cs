using System;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PythonProcessing.DataLayer
{
	public class AdoNetDb : IDataLayer
	{
		private const string _connectionString = "N/A";

		public void ExecuteSQLBulkCopyInsert(string destinationTable, List<string> columnMappings, DataTable dtRowsToInsert)
		{
			using (SqlBulkCopy sbc = new SqlBulkCopy(_connectionString))
			{
				sbc.DestinationTableName = destinationTable;
				sbc.BatchSize = dtRowsToInsert.Rows.Count;

				// Map the source column from dtInsertRows table to the destination columns in SQL Server 2014 data table
				foreach (var columnMapping in columnMappings)
				{
					sbc.ColumnMappings.Add(columnMapping, columnMapping);
				}

				sbc.WriteToServer(dtRowsToInsert);
			}
		}

		public DataTable PopulateDataTable(string destinationTable, Dictionary<string, Type> columnMappings, List<string> content)
		{
			DataTable dt = new DataTable();

			foreach (var columnMapping in columnMappings)
			{
				dt.Columns.Add(columnMapping.Key, columnMapping.Value);
			}

			foreach (var codeReview in content)
			{
				dt.Rows.Add(codeReview);
			}

			return dt;
		}
	}
}
