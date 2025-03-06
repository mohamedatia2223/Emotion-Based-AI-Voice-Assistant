CREATE TABLE [dbo].[owners](
	[id] [int] IDENTITY(1,1) PRIMARY KEY NOT NULL,
	[name] [nvarchar](100) NOT NULL,
	[age] [int] NOT NULL,
	[image_Base64] [nvarchar](Max) NOT NULL
	)
